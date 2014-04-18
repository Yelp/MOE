# -*- coding: utf-8 -*-
#!/usr/bin/python

# Copyright (C) 2012 Scott Clark. All rights reserved.

#import pdb # debugger

#import logging # logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
#from optparse import OptionParser # parser = OptionParser()

#import numpy # for sci comp
#import matplotlib.pylab as plt # for plotting

import time # for timing
#import commands # print commands.getoutput(script)

# pyCUDA
import pycuda.autoinit
import pycuda.curandom
import pycuda.driver as drv
import numpy
from pycuda.compiler import SourceModule

import GPP_math
import GPP_plotter
import GaussianProcessPrior as GP

def uglify(code):
    code = code.split('\n')
    for i,line in enumerate(code):
        code[i] = line.strip()
        if code[i][:2] == '//':
            code[i] = ''
    return '\n'.join(code)

def cuda_get_next_points(GPP, points_to_sample=None, n_cores=1, total_restarts=8, grad_EI_its=100, pr_steps=10):
    print
    mod = SourceModule(uglify("""
        #include <curand_kernel.h>

        static const float ALPHA = 1.0;
        static const float P = 2.0;
        static const float L = 2.0;

        extern "C"
        {
            __device__ float one_norm(float *point_one, float *point_two, int size){
                // calculates the one norm of two vectors (point_one and point_two of size size)
                float norm = 0.0;
                size_t i;
                for(i=0; i<size; i++){
                    norm += pow(point_one[i] - point_two[i], P);
                }
                return sqrt(norm);
            }

            __device__ float covariance(float *point_one, float *point_two, int size){
                float beta = -1.0/(2.0*L);
                return ALPHA*exp(beta*pow(one_norm(point_one, point_two, size), P));
            }

            __device__ float grad_covariance(float *point_one, float *point_two, int size, int ind){
                float base = (P/(2.0*L))*pow(one_norm(point_one, point_two, size),P-2.0f)*covariance(point_one, point_two, size);
                return (point_two[ind] - point_one[ind])*base;
            }

            __device__ void destroy_matrix(float **matrix, int size_of_matrix){
                // free a malloc'ed matrix
                size_t i;
                for(i=0; i<size_of_matrix; i++){
                    free(matrix[i]);
                }
                free(matrix);
            }



            __device__ void inverse_via_backward_sub(float **inv_mat, float **matrix, int size_of_matrix){
                // get inverse of a lower triangular matrix via backwards substitution
                size_t inv_col, inv_row, back_col;
                for(inv_col=0; inv_col<size_of_matrix; inv_col++){
                    for(inv_row=0; inv_row<size_of_matrix; inv_row++){
                        if(inv_col == inv_row){
                            inv_mat[inv_row][inv_col] = 1.0;
                        }else{
                            inv_mat[inv_row][inv_col] = 0.0;
                        }
                        for(back_col=0; back_col<inv_row; back_col++){
                            inv_mat[inv_row][inv_col] -= matrix[inv_row][back_col]*inv_mat[back_col][inv_col];
                        }
                        inv_mat[inv_row][inv_col] /= matrix[inv_row][inv_row];
                    }
                }
            }

            __device__ void matrix_product(float **prod_mat, float **matrix_one, float **matrix_two, int left_dim, int mid_dim, int right_dim){
                // finds the product of two square matricies
                float sum;
                size_t i,j,k;
                for(i=0; i<left_dim; i++){
                    for(j=0; j<right_dim; j++){
                        sum = 0.0;
                        for(k=0; k<mid_dim; k++){
                            sum += matrix_one[i][k]*matrix_two[k][j];
                        }
                        prod_mat[i][j] = sum;
                    }
                }
            }

            __device__ void matrix_transpose(float **matrix_T, float **matrix, int row_size, int col_size){
                // returns the transpose of a matrix
                size_t i,j;
                for(i=0; i<col_size; i++){
                    for(j=0; j<row_size; j++){
                        matrix_T[i][j] = matrix[j][i];
                    }
                }
            }

            __device__ void build_covariance_matrix(float **cov_mat, float **points_sampled, int dim_of_points, int size_of_matrix){
                // calculate the covariance matrix defined in GPP_cov.h
                size_t i,j;
                for(i=0; i<size_of_matrix; i++){
                    for(j=0; j<size_of_matrix; j++){
                        cov_mat[i][j] = covariance(points_sampled[i], points_sampled[j], dim_of_points);
                    }
                }
            }

            __device__ void build_mix_covariance_matrix(float **cov_mat, float **points_to_sample, float **points_sampled, int dim_of_points, int size_of_to_sample, int size_of_sampled){
                // calculate the covariance matrix defined in GPP_cov.h
                size_t i,j;
                for(i=0; i<size_of_to_sample; i++){
                    for(j=0; j<size_of_sampled; j++){
                        cov_mat[i][j] = covariance(points_to_sample[i], points_sampled[j], dim_of_points);
                    }
                }
            }

            __device__ void matrix_vector_multiply(float *answer, float **matrix, float *vector, int matrix_row_size, int matrix_col_size){
                size_t i,j;
                float sum;
                for(i=0; i<matrix_row_size; i++){
                    sum = 0.0;
                    for(j=0; j<matrix_col_size; j++){
                        sum += matrix[i][j]*vector[j];
                    }
                    answer[i] = sum;
                }
            }

            __device__ void matrix_subtraction(float **answer, float **matrix_one, float **matrix_two, int row_size, int col_size){
                size_t i,j;
                for(i=0; i<row_size; i++){
                    for(j=0; j<col_size; j++){
                        answer[i][j] = matrix_one[i][j] - matrix_two[i][j];
                    }
                }
            }

            __device__ void get_cholesky_alpha(float *cholesky_alpha, float **points_sampled, float *points_sampled_value, int dim_of_points, int size, float **K_inv){
                // See RW pg 19 alg 2.1
                matrix_vector_multiply(cholesky_alpha, K_inv, points_sampled_value, size, size);
            }

            __device__ void special_matrix_vector_multiply(float **answer, float ***matrix, float *vector, int dim_one, int dim_two, int dim_three){
                float sum = 0.0;
                size_t i,j,k;
                for(i=0; i<dim_one; i++){
                    for(k=0; k<dim_three; k++){
                        sum = 0.0;
                        for(j=0; j<dim_two; j++){
                            sum += matrix[i][j][k]*vector[j];
                        }
                        answer[i][k] = sum;
                    }
                }
            }

            __device__ void find_cholesky_decomp(float **cholesky_decomp, float **matrix, int size_of_matrix){
                // find the cholesky decomposition of a positive definite matrix
                // see smith 1995 or Golub, Van Loan 1983

                // make lower triangular
                size_t i,j,k;
                for(i=0; i<size_of_matrix; i++){
                    for(j=0; j<size_of_matrix; j++){
                        if(i < j){ // upper half
                            cholesky_decomp[i][j] = 0.0;
                        }else{ // lower half and diagonal
                            cholesky_decomp[i][j] = matrix[i][j];
                        }
                    }
                }

                // apply algorithm O(n^3)
                for(k=0; k<size_of_matrix; k++){
                    if(cholesky_decomp[k][k] > 1e-8){
                        cholesky_decomp[k][k] = sqrt(cholesky_decomp[k][k]);
                        for(j=k+1; j<size_of_matrix; j++){
                            cholesky_decomp[j][k] = cholesky_decomp[j][k]/cholesky_decomp[k][k];
                        }
                        for(j=k+1; j<size_of_matrix; j++){
                            for(i=j; i<size_of_matrix; i++){
                                cholesky_decomp[i][j] = cholesky_decomp[i][j] - cholesky_decomp[i][k]*cholesky_decomp[j][k];
                            }
                        }
                    }
                }
            }

            __device__ void get_grad_var(float ***grad_cholesky_decomp, float **points_to_sample, float **points_sampled, float *points_sampled_value, int dim_of_points, int size_of_to_sample, int size_of_sampled, int var_of_grad, float **K_inv, float **cholesky_decomp, float *grad_cov){
                // find the grad of a cholesky decomposition of a positive definite matrix
                // wrt the var_of_grad see smith 1995

                // step 1 of appendix 2
                size_t i,j,k,l,m,d;
                for(i=0; i<size_of_to_sample; i++){
                    for(j=0; j<size_of_to_sample; j++){
                        if(var_of_grad == i){
                            for(d=0; d<dim_of_points; d++){
                                grad_cholesky_decomp[i][j][d] = grad_covariance(points_to_sample[i], points_to_sample[j], dim_of_points, d);
                            }
                            for(k=0; k<size_of_sampled; k++){
                                for(l=0; l<size_of_sampled; l++){
                                    for(d=0; d<dim_of_points; d++){
                                        grad_cov[d] = grad_covariance(points_to_sample[i], points_sampled[l], dim_of_points, d);
                                    }
                                    for(m=0; m<dim_of_points; m++){
                                        grad_cholesky_decomp[i][j][m] -= K_inv[l][k]*covariance(points_to_sample[j], points_sampled[k], dim_of_points)*grad_cov[m];
                                    }
                                }
                            }
                        }else if(var_of_grad == j){
                            for(d=0; d<dim_of_points; d++){
                                grad_cholesky_decomp[i][j][d] = grad_covariance(points_to_sample[j], points_to_sample[i], dim_of_points, d);
                            }
                            for(k=0; k<size_of_sampled; k++){
                                for(l=0; l<size_of_sampled; l++){
                                    for(d=0; d<dim_of_points; d++){
                                        grad_cov[d] = grad_covariance(points_to_sample[j], points_sampled[k], dim_of_points, d);
                                    }
                                    for(m=0; m<dim_of_points; m++){
                                        grad_cholesky_decomp[i][j][m] -= K_inv[l][k]*covariance(points_to_sample[i], points_sampled[l], dim_of_points)*grad_cov[m];
                                    }
                                }
                            }
                        }else{
                            for(m=0; m<dim_of_points; m++){
                                grad_cholesky_decomp[i][j][m] = 0.0;
                            }
                        }
                    }
                }

                // zero out upper half of the matrix
                for(i=0; i<size_of_to_sample; i++){
                    for(j=0; j<size_of_to_sample; j++){
                        if(i < j){
                            cholesky_decomp[i][j] = 0.0;
                            for(m=0; m<dim_of_points; m++){
                                grad_cholesky_decomp[i][j][m] = 0.0;
                            }
                        }
                    }
                }

                for(k=0; k<size_of_to_sample; k++){
                    if(fabs(cholesky_decomp[k][k]) > 1e-8){
                        cholesky_decomp[k][k] = sqrt(fabs(cholesky_decomp[k][k]));
                        for(m=0; m<dim_of_points; m++){
                            grad_cholesky_decomp[k][k][m] = 0.5*grad_cholesky_decomp[k][k][m]/cholesky_decomp[k][k];
                        }
                        for(j=k+1; j<size_of_to_sample; j++){
                            cholesky_decomp[j][k] = cholesky_decomp[j][k]/cholesky_decomp[k][k];
                            for(m=0; m<dim_of_points; m++){
                                grad_cholesky_decomp[j][k][m] = (grad_cholesky_decomp[j][k][m] + cholesky_decomp[j][k]*grad_cholesky_decomp[k][k][m])/cholesky_decomp[k][k];
                            }
                        }
                        for(j=k+1; j<size_of_to_sample; j++){
                            for(i=j; i<size_of_to_sample; i++){
                                cholesky_decomp[i][j] = cholesky_decomp[i][j] - cholesky_decomp[i][k]*cholesky_decomp[j][k];
                                for(m=0; m<dim_of_points; m++){
                                    grad_cholesky_decomp[i][j][m] = grad_cholesky_decomp[i][j][m] - grad_cholesky_decomp[i][k][m]*cholesky_decomp[j][k] - cholesky_decomp[i][k]*grad_cholesky_decomp[j][k][m];
                                }
                            }
                        }
                    }
                }
            }

            __device__ float euclidian_norm(float *point, int dim_of_point){
                float agg = 0.0;
                size_t i;
                for(i=0; i<dim_of_point; i++){
                    agg += pow(point[i], 2.0f);
                }
                return sqrt(agg);
            }

            __device__ void get_expected_grad_EI(float *exp_grad_EI, float **points_to_sample, float **points_sampled, float *points_sampled_value, int dim_of_points, int size_of_to_sample, int size_of_sampled, float best_so_far, int exp_grad_its, float *normals, float *to_sample_mean, float **grad_mu, float **to_sample_var, float **cholesky_decomp, float ***grad_chol_decomp, float *aggrigate, float **K, float **K_inv, float **K_star, float **K_star_T, float **K_star_star, float **L, float **L_inv, float **L_inv_T, float *cholesky_alpha, float **v, float **v_T, float **vTv, float *pre_vector, float *grad_cov, float *EI_this_step_from_var)
            {
                int i,j,k;

                for(j=0; j<dim_of_points; j++){
                    exp_grad_EI[j] = 0.0;
                }

                build_covariance_matrix(K_star_star, points_to_sample, dim_of_points, size_of_to_sample);

                for(i=0; i<size_of_to_sample; i++){
                    for(k=0; k<dim_of_points; k++){grad_mu[i][k] = 0.0;}
                    for(j=0; j<size_of_sampled; j++){
                        for(k=0; k<dim_of_points; k++){
                            grad_mu[i][k] += grad_covariance(points_to_sample[i], points_sampled[j], dim_of_points, k)*pre_vector[j];
                        }
                    }
                }

                build_mix_covariance_matrix(K_star, points_to_sample, points_sampled, dim_of_points, size_of_to_sample, size_of_sampled);
                matrix_transpose(K_star_T, K_star, size_of_to_sample, size_of_sampled);
                matrix_product(v, L_inv, K_star_T, size_of_sampled, size_of_sampled, size_of_to_sample);
                matrix_transpose(v_T, v, size_of_sampled, size_of_to_sample);
                matrix_product(vTv, v_T, v, size_of_to_sample, size_of_sampled, size_of_to_sample);

                matrix_vector_multiply(to_sample_mean, K_star, cholesky_alpha, size_of_to_sample, size_of_sampled);

                matrix_subtraction(to_sample_var, K_star_star, vTv, size_of_to_sample, size_of_to_sample);

                find_cholesky_decomp(cholesky_decomp, to_sample_var, size_of_to_sample);

                get_grad_var(grad_chol_decomp, points_to_sample, points_sampled, points_sampled_value, dim_of_points, size_of_to_sample, size_of_sampled, 0, K_inv, to_sample_var, grad_cov);

                for(j=0; j<dim_of_points; j++){
                    aggrigate[j] = 0.0;
                }

                float improvement_this_step;
                int winner = -1;
                float EI_total;

                // RNG setup
                const int idx = threadIdx.x;
                unsigned int seed = idx;
                curandState s;
                // seed a random number generator
                curand_init(seed, 0, 0, &s);

                for(i=0; i<exp_grad_its; i++){

                    improvement_this_step = 0.0;
                    for(j=0; j<size_of_to_sample; j++){
                        normals[j] = curand_normal(&s);
                    }

                    for(j=0; j<size_of_to_sample; j++){
                        EI_this_step_from_var[j] = 0.0;
                        for(k=j; k<size_of_to_sample; k++){
                            EI_this_step_from_var[j] += cholesky_decomp[k][j]*normals[k];
                        }
                    }

                    for(j=0; j<size_of_to_sample; j++){
                        EI_total = best_so_far - (to_sample_mean[j] + EI_this_step_from_var[j]);
                        if(EI_total > improvement_this_step){
                            improvement_this_step = EI_total;
                            winner = j;
                        }
                    }


                    if(improvement_this_step > 0.0){

                        if(winner == 0){
                            for(j=0; j<dim_of_points; j++){
                                aggrigate[j] -= grad_mu[0][j];
                            }
                        }

                        for(j=0; j<dim_of_points; j++){
                            for(k=0; k<size_of_to_sample; k++){
                                aggrigate[j] -= grad_chol_decomp[winner][k][j]*normals[k];
                            }
                        }
                    }
                }
                for(j=0; j<dim_of_points; j++){
                    exp_grad_EI[j] = aggrigate[j]/(float)exp_grad_its;
                }
            }

            __global__ void get_next_points(float *in_current_point, float *in_points_to_sample, float *in_points_sampled, float *points_sampled_value, int dim_of_points, int size_of_current, int size_of_to_sample, int size_of_sampled, float best_so_far, int max_steps, int max_int_steps, float gamma, float pre_mult, int step_on, int number_of_sim_cores)
            {
                // alocate all memory
                int i, j, k;
                const int idx = threadIdx.x;
                float alpha_n;
                const int core_block_start = number_of_sim_cores*(idx/number_of_sim_cores);

                const int var_on = number_of_sim_cores*(idx/number_of_sim_cores) + idx%number_of_sim_cores;

                size_of_to_sample += number_of_sim_cores;

                float **points_to_sample = (float **)malloc(sizeof(float*)*(size_of_to_sample));
                // put in the current point for this CUDA core, grad will be taken wrt this point
                points_to_sample[0] = (float *)malloc(sizeof(float)*dim_of_points);
                for(j=0; j<dim_of_points; j++){
                    points_to_sample[0][j] = in_current_point[idx + j*size_of_current];
                }
                // destroy_matrix(points_to_sample, size_of_to_sample);

                // put in the rest of the points in the simulated cores (these change each step)
                k = 1;
                for(i=0; i<number_of_sim_cores; i++){
                    if(core_block_start + i != idx){ // we already have this point
                        points_to_sample[k] = (float *)malloc(sizeof(float)*dim_of_points);
                        for(j=0; j<dim_of_points; j++){
                            points_to_sample[k][j] = in_current_point[core_block_start + i + j*size_of_current];
                        }
                        k++;
                    }
                }

                // put in the points that are being sampled by black box function (outside experiment). These do not change
                for(i=number_of_sim_cores; i<size_of_to_sample; i++){
                    points_to_sample[i] = (float *)malloc(sizeof(float)*dim_of_points);
                    for(j=0; j<dim_of_points; j++){
                        points_to_sample[i][j] = in_points_to_sample[i - number_of_sim_cores + j*size_of_to_sample];
                    }
                }


                float *exp_grad_EI = (float *)malloc(sizeof(float)*dim_of_points);
                // free(exp_grad_EI);

                float **points_sampled = (float **)malloc(sizeof(float *)*size_of_sampled);
                for(i=0; i<size_of_sampled; i++){
                    points_sampled[i] = (float *)malloc(sizeof(float)*dim_of_points);
                    for(j=0; j<dim_of_points; j++){
                        points_sampled[i][j] = in_points_sampled[i + j*size_of_sampled];
                    }
                }
                // destroy_matrix(points_sampled, size_of_sampled);

                float *normals = (float *)malloc(sizeof(float)*(size_of_to_sample));
                // free(normals);
                float *to_sample_mean = (float *)malloc(sizeof(float)*(size_of_to_sample));
                // free(to_sample_mean);

                float **K = (float **)malloc(sizeof(float*)*size_of_sampled);
                for(i=0; i<size_of_sampled; i++){
                    K[i] = (float *)malloc(sizeof(float)*size_of_sampled);
                }
                // destroy_matrix(K, size_of_sampled);
                float **K_inv = (float **)malloc(sizeof(float*)*size_of_sampled);
                for(i=0; i<size_of_sampled; i++){
                    K_inv[i] = (float *)malloc(sizeof(float)*size_of_sampled);
                }
                // destroy_matrix(K_inv, size_of_sampled);

                float **K_star = (float **)malloc(sizeof(float*)*size_of_to_sample);
                for(i=0; i<size_of_to_sample; i++){
                    K_star[i] = (float *)malloc(sizeof(float)*size_of_sampled);
                }
                // destroy_matrix(K_star, size_of_to_sample);
                float **K_star_T = (float **)malloc(sizeof(float*)*size_of_sampled);
                for(i=0; i<size_of_sampled; i++){
                    K_star_T[i] = (float *)malloc(sizeof(float)*size_of_to_sample);
                }
                // destroy_matrix(K_star_T, size_of_sampled);

                float **K_star_star = (float **)malloc(sizeof(float*)*size_of_to_sample);
                for(i=0; i<size_of_to_sample; i++){
                    K_star_star[i] = (float *)malloc(sizeof(float)*size_of_to_sample);
                }
                // destroy_matrix(K_star_star, size_of_to_sample);

                /*
                float ***grad_K_star = (float ***)malloc(sizeof(float**)*size_of_to_sample);
                for(i=0; i<size_of_to_sample; i++){
                    grad_K_star[i] = (float **)malloc(sizeof(float*)*size_of_sampled);
                    for(j=0; j<size_of_sampled; j++){
                        grad_K_star[i][j] = (float *)malloc(sizeof(float)*dim_of_points);
                    }
                }
                */

                float **L = (float **)malloc(sizeof(float*)*size_of_sampled);
                for(i=0; i<size_of_sampled; i++){
                    L[i] = (float *)malloc(sizeof(float)*size_of_sampled);
                }
                // destroy_matrix(L, size_of_sampled);
                float **L_inv = (float **)malloc(sizeof(float*)*size_of_sampled);
                for(i=0; i<size_of_sampled; i++){
                    L_inv[i] = (float *)malloc(sizeof(float)*size_of_sampled);
                }
                // destroy_matrix(L_inv, size_of_sampled);
                float **L_inv_T = (float **)malloc(sizeof(float*)*size_of_sampled);
                for(i=0; i<size_of_sampled; i++){
                    L_inv_T[i] = (float *)malloc(sizeof(float)*size_of_sampled);
                }
                // destroy_matrix(L_inv_T, size_of_sampled);

                float **v = (float **)malloc(sizeof(float*)*size_of_sampled);
                for(i=0; i<size_of_sampled; i++){
                    v[i] = (float *)malloc(sizeof(float)*size_of_to_sample);
                }
                // destroy_matrix(v, size_of_sampled);
                float **v_T = (float **)malloc(sizeof(float*)*size_of_to_sample);
                for(i=0; i<size_of_to_sample; i++){
                    v_T[i] = (float *)malloc(sizeof(float)*size_of_sampled);
                }
                // destroy_matrix(v_T, size_of_to_sample);
                float **vTv = (float **)malloc(sizeof(float*)*size_of_to_sample);
                for(i=0; i<size_of_to_sample; i++){
                    vTv[i] = (float *)malloc(sizeof(float)*size_of_to_sample);
                }
                // destroy_matrix(vTv, size_of_to_sample);

                float **grad_mu = (float **)malloc(sizeof(float*)*size_of_to_sample);
                for(i=0; i<size_of_to_sample; i++){
                    grad_mu[i] = (float *)malloc(sizeof(float)*dim_of_points);
                }
                // destroy_matrix(grad_mu, size_of_to_sample);
                float **to_sample_var = (float **)malloc(sizeof(float*)*size_of_to_sample);
                for(i=0; i<size_of_to_sample; i++){
                    to_sample_var[i] = (float *)malloc(sizeof(float)*size_of_to_sample);
                }
                // destroy_matrix(to_sample_var, size_of_to_sample);
                float **cholesky_decomp = (float **)malloc(sizeof(float*)*size_of_to_sample);
                for(i=0; i<size_of_to_sample; i++){
                    cholesky_decomp[i] = (float *)malloc(sizeof(float)*size_of_to_sample);
                }
                // destroy_matrix(cholesky_decomp, size_of_to_sample);

                float *cholesky_alpha = (float *)malloc(sizeof(float)*size_of_sampled);
                // free(cholesky_alpha);
                float *pre_vector = (float *)malloc(sizeof(float)*size_of_sampled);
                // free(pre_vector);
                float *aggrigate = (float *)malloc(sizeof(float)*dim_of_points);
                // free(aggrigate);
                float *grad_cov = (float *)malloc(sizeof(float)*dim_of_points);
                // free(grad_cov);
                float *EI_this_step_from_var = (float *)malloc(sizeof(float)*size_of_to_sample+1);
                // free(EI_this_step_from_var);

                float ***grad_chol_decomp = (float ***)malloc(sizeof(float**)*size_of_to_sample);
                for(i=0; i<size_of_to_sample; i++){
                    grad_chol_decomp[i] = (float **)malloc(sizeof(float*)*size_of_to_sample);
                    for(j=0; j<size_of_to_sample; j++){
                        grad_chol_decomp[i][j] = (float *)malloc(sizeof(float)*dim_of_points);
                    }
                }

                build_covariance_matrix(K, points_sampled, dim_of_points, size_of_sampled); // ind

                find_cholesky_decomp(L, K, size_of_sampled); // ind
                inverse_via_backward_sub(L_inv, L, size_of_sampled); // ind
                matrix_transpose(L_inv_T, L_inv, size_of_sampled, size_of_sampled); // ind

                matrix_product(K_inv, L_inv_T, L_inv, size_of_sampled, size_of_sampled, size_of_sampled); // ind

                get_cholesky_alpha(cholesky_alpha, points_sampled, points_sampled_value, dim_of_points, size_of_sampled, K_inv); // ind

                matrix_vector_multiply(pre_vector, K_inv, points_sampled_value, size_of_sampled, size_of_sampled); // ind

                // update points
                for(i=0; i<max_steps; i++){

                    alpha_n = pre_mult*pow((float)(i+step_on+1), -gamma);

                    get_expected_grad_EI(exp_grad_EI, points_to_sample, points_sampled, points_sampled_value, dim_of_points, size_of_to_sample, size_of_sampled, best_so_far, max_int_steps,  normals, to_sample_mean, grad_mu, to_sample_var, cholesky_decomp, grad_chol_decomp, aggrigate, K, K_inv, K_star, K_star_T, K_star_star, L, L_inv, L_inv_T, cholesky_alpha, v, v_T, vTv, pre_vector, grad_cov, EI_this_step_from_var);

                    for(j=0; j<dim_of_points; j++){
                        points_to_sample[0][j] += alpha_n*exp_grad_EI[j];
                    }
                }

                for(j=0; j<dim_of_points; j++){
                    in_current_point[idx + j*size_of_current] = points_to_sample[0][j];
                }

                // free memory
                destroy_matrix(points_to_sample, size_of_to_sample);
                free(exp_grad_EI);
                destroy_matrix(points_sampled, size_of_sampled);
                free(normals);
                free(to_sample_mean);
                destroy_matrix(K, size_of_sampled);
                destroy_matrix(K_inv, size_of_sampled);
                destroy_matrix(K_star, size_of_to_sample);
                destroy_matrix(K_star_T, size_of_sampled);
                destroy_matrix(K_star_star, size_of_to_sample);
                /*
                for(i=0; i<size_of_to_sample; i++){
                    destroy_matrix(grad_K_star[i], size_of_sampled);
                }
                free(grad_K_star);
                */
                destroy_matrix(L, size_of_sampled);
                destroy_matrix(L_inv, size_of_sampled);
                destroy_matrix(L_inv_T, size_of_sampled);
                destroy_matrix(v, size_of_sampled);
                destroy_matrix(v_T, size_of_to_sample);
                destroy_matrix(vTv, size_of_to_sample);
                destroy_matrix(grad_mu, size_of_to_sample);
                destroy_matrix(to_sample_var, size_of_to_sample);
                destroy_matrix(cholesky_decomp, size_of_to_sample);
                free(cholesky_alpha);
                free(pre_vector);
                free(aggrigate);
                free(grad_cov);
                free(EI_this_step_from_var);
                for(i=0; i<size_of_to_sample; i++){
                    destroy_matrix(grad_chol_decomp[i], size_of_to_sample);
                }
                free(grad_chol_decomp);
            }
        }
        """),no_extern_c=True)

    get_exp_next_point = mod.get_function("get_next_points")

    # GPP, points_to_sample=None, n_cores=2, total_restarts=100, grad_EI_its=1000, pr_steps=100

    max_block = 8
    num_small_blocks = max_block/n_cores
    big_restarts = total_restarts/num_small_blocks
    # starting points
    current_points = []

    def flatten(ins):
        new = []
        for i in range(len(ins[0])):
            for tup in ins:
                new.append(tup[i])
        return new

    # points_sampled / value
    points_sampled = []
    for ps in GPP.points_sampled:
        points_sampled.append(numpy.array(ps.point, dtype=numpy.float32))
    print points_sampled
    points_sampled = numpy.array(points_sampled, dtype=numpy.float32)
    points_sampled_value = numpy.array(GPP.values_of_samples, dtype=numpy.float32)
    size_of_sampled = numpy.int32(len(GPP.points_sampled))
    points_sampled = numpy.array(flatten(points_sampled), dtype=numpy.float32)

    # points_to_sample
    if not points_to_sample:
        points_to_sample = numpy.array([numpy.array([100,100], dtype=numpy.float32)])
    size_of_to_sample = numpy.int32(len(points_to_sample))
    points_to_sample = numpy.array(flatten(points_to_sample), dtype=numpy.float32)

    # other input
    dim_of_points = numpy.int32(len(GPP.points_sampled[0].point))

    best_so_far = numpy.float32(GPP.best_so_far)
    max_steps = numpy.int32(1)
    max_int_steps = numpy.int32(grad_EI_its)
    # pk params
    gamma = numpy.float32(0.8)
    pre_mult = numpy.float32(1.0)

    for big_restart in range(big_restarts):
        # starting points sampled from latin hypercube
        current_points = []
        for i in range(num_small_blocks):
            current_points.extend(GPP_math.get_latin_hypercube_points(n_cores, GPP.domain))
        current_points = numpy.array(current_points, dtype=numpy.float32)
        size_of_current = numpy.int32(len(current_points))
        current_points = numpy.array(flatten(current_points), dtype=numpy.float32)

        new_points = []
        for i in range(size_of_current):
            tmp = []
            for j in range(dim_of_points):
                tmp.append(current_points[i + j*size_of_current])
            new_points.append(numpy.array(tmp))

        new_points = numpy.array(new_points)
        print "start"
        print new_points


        for step_on in range(pr_steps):

            get_exp_next_point(
                drv.InOut(current_points),
                drv.In(points_to_sample),
                drv.In(points_sampled),
                drv.In(points_sampled_value),
                dim_of_points, size_of_current, size_of_to_sample, size_of_sampled, best_so_far, max_steps,
                max_int_steps, gamma, pre_mult, numpy.int32(step_on), numpy.int32(n_cores),
                block=(int(size_of_current),1,1), grid=(1,1))

        #unflatten
        new_points = []
        for i in range(size_of_current):
            tmp = []
            for j in range(dim_of_points):
                tmp.append(current_points[i + j*size_of_current])
            new_points.append(numpy.array(tmp))

        new_points = numpy.array(new_points)
        print "big step taken"
        print new_points
        #print cuda_get_exp_EI(GPP, new_points)

    return new_points


def cuda_get_next_points_new(GPP, points_being_sampled=None, n_cores=4, max_block_size=64, total_restarts=1, grad_EI_its=100000, pr_steps=5, paths_only=False, gamma=0.8, pre_mult=1):
    mod = SourceModule(uglify("""
        #include <curand_kernel.h>
        extern "C"
        {
            __device__ void get_chol_and_grad_chol(float **cholesky_decomp, float ***grad_cholesky_decomp, const int dim_of_points, const int size_of_to_sample){
                // ASSUME cholesky_decomp = sigma and grad_cholesky_decomp = grad_sigma (step 1)
                int i, j, k, m;

                // zero out upper half of the matrix
                for(i=0; i<size_of_to_sample; i++){
                    for(j=0; j<size_of_to_sample; j++){
                        if(i < j){
                            cholesky_decomp[i][j] = 0.0;
                            for(m=0; m<dim_of_points; m++){
                                grad_cholesky_decomp[i][j][m] = 0.0;
                            }
                        }
                    }
                }

                // step 2
                for(k=0; k<size_of_to_sample; k++){
                    if(fabs(cholesky_decomp[k][k]) > 1e-8){
                        // step 2a
                        cholesky_decomp[k][k] = sqrt(fabs(cholesky_decomp[k][k]));
                        for(m=0; m<dim_of_points; m++){
                            grad_cholesky_decomp[k][k][m] = 0.5*grad_cholesky_decomp[k][k][m]/cholesky_decomp[k][k];
                        }
                        // step 2b
                        for(j=k+1; j<size_of_to_sample; j++){
                            cholesky_decomp[j][k] = cholesky_decomp[j][k]/cholesky_decomp[k][k];
                            for(m=0; m<dim_of_points; m++){
                                grad_cholesky_decomp[j][k][m] = (grad_cholesky_decomp[j][k][m] + cholesky_decomp[j][k]*grad_cholesky_decomp[k][k][m])/cholesky_decomp[k][k];
                            }
                        }
                        // step 2c
                        for(j=k+1; j<size_of_to_sample; j++){
                            for(i=j; i<size_of_to_sample; i++){
                                cholesky_decomp[i][j] = cholesky_decomp[i][j] - cholesky_decomp[i][k]*cholesky_decomp[j][k];
                                for(m=0; m<dim_of_points; m++){
                                    grad_cholesky_decomp[i][j][m] = grad_cholesky_decomp[i][j][m] - grad_cholesky_decomp[i][k][m]*cholesky_decomp[j][k] - cholesky_decomp[i][k]*grad_cholesky_decomp[j][k][m];
                                }
                            }
                        }
                    }
                }
            }

            __global__ void get_grad_EI(float *exp_grad_EI, float *to_sample_mean_in, float *grad_mu_in, float *cholesky_decomp_sample_var, float *grad_chol_decomp_sample_var_in, const int dim_of_points, const int size_of_to_sample, const float best_so_far, const int exp_grad_its, const int number_of_sim_cores, const unsigned int rand_seed)
            {
                int i,j,k;

                const int idx = threadIdx.x;

                const int core_block_start = idx/number_of_sim_cores;
                const int two_shift = core_block_start*size_of_to_sample*size_of_to_sample;
                const int three_shift = core_block_start*size_of_to_sample*size_of_to_sample*dim_of_points;

                const int var_on = idx%number_of_sim_cores;

                // allocate memory for mu
                float *mu = (float *)malloc(sizeof(float)*size_of_to_sample);
                // set values from to_sample_mean
                for(i=0; i<size_of_to_sample; i++){
                    mu[i] = to_sample_mean_in[i + core_block_start*size_of_to_sample];
                }

                // allocate memory for grad_mu
                float *grad_mu = (float *)malloc(sizeof(float)*dim_of_points);
                // set values from grad_mu_in
                for(j=0; j<dim_of_points; j++){
                    grad_mu[j] = grad_mu_in[j + var_on*dim_of_points + core_block_start*size_of_to_sample*dim_of_points];
                }

                // allocate memory for sigma (soon to be cholesky(sigma))
                float **cholesky_decomp = (float **)malloc(sizeof(float*)*size_of_to_sample);
                for(i=0; i<size_of_to_sample; i++){
                    cholesky_decomp[i] = (float *)malloc(sizeof(float)*size_of_to_sample);
                }
                // set values from cholesky_decomp_sample_var
                for(i=0; i<size_of_to_sample; i++){
                    for(j=0; j<size_of_to_sample; j++){
                        cholesky_decomp[i][j] = cholesky_decomp_sample_var[i + j*size_of_to_sample + two_shift];
                    }
                }

                // allocate memory for grad_sigma (soon to be grad_cholesky(sigma))
                float ***grad_chol_decomp = (float ***)malloc(sizeof(float**)*size_of_to_sample);
                for(i=0; i<size_of_to_sample; i++){
                    grad_chol_decomp[i] = (float **)malloc(sizeof(float*)*size_of_to_sample);
                    for(j=0; j<size_of_to_sample; j++){
                        grad_chol_decomp[i][j] = (float *)malloc(sizeof(float)*dim_of_points);
                    }
                }
                // set values from grad_chol_decomp_sample_var
                // this is non-trivial so pay attention
                k=0;
                for(i=0; i<size_of_to_sample; i++){
                    for(j=0; j<size_of_to_sample; j++){
                        for(k=0; k<dim_of_points; k++){
                            if(var_on == i){
                                grad_chol_decomp[i][j][k] = grad_chol_decomp_sample_var_in[k + j*dim_of_points + 0*dim_of_points*size_of_to_sample + var_on*dim_of_points*size_of_to_sample + three_shift];
                            }else if(var_on == j){
                                grad_chol_decomp[i][j][k] = grad_chol_decomp_sample_var_in[k + i*dim_of_points + 0*dim_of_points*size_of_to_sample + var_on*dim_of_points*size_of_to_sample + three_shift];
                            }else{
                                grad_chol_decomp[i][j][k] = 0.0;
                            }
                        }
                    }
                }

                // calculate cholesky decompositions in place
                get_chol_and_grad_chol(cholesky_decomp, grad_chol_decomp, dim_of_points, size_of_to_sample);

                // allocate memory for MC steps
                float *aggrigate = (float *)malloc(sizeof(float)*dim_of_points);
                for(j=0; j<dim_of_points; j++){
                    aggrigate[j] = 0.0;
                }
                float *normals = (float *)malloc(sizeof(float)*size_of_to_sample);
                for(j=0; j<size_of_to_sample; j++){
                    normals[j] = 0.0;
                }
                float *EI_this_step_from_var = (float *)malloc(sizeof(float)*size_of_to_sample);
                for(j=0; j<dim_of_points; j++){
                    EI_this_step_from_var[j] = 0.0;
                }

                float improvement_this_step;
                int winner = -1;
                float EI_total;

                // RNG setup
                unsigned int seed = rand_seed;
                curandState s;
                // seed a random number generator
                curand_init(seed, idx, idx*exp_grad_its*size_of_to_sample, &s);

                // MC estimation of grad_EI
                for(i=0; i<exp_grad_its; i++){

                    for(j=0; j<size_of_to_sample; j++){
                        normals[j] = curand_normal(&s);
                    }

                    // find the winner and change aggrigate
                    improvement_this_step = 0.0;
                    for(j=0; j<size_of_to_sample; j++){
                        EI_this_step_from_var[j] = 0.0;
                        for(k=0; k<size_of_to_sample; k++){
                            EI_this_step_from_var[j] += cholesky_decomp[j][k]*normals[k];
                        }
                    }

                    for(j=0; j<size_of_to_sample; j++){
                        EI_total = best_so_far - mu[j] - EI_this_step_from_var[j];
                        if(EI_total > improvement_this_step){
                            improvement_this_step = EI_total;
                            winner = j;
                        }
                    }

                    if(improvement_this_step > 0.0){

                        if(winner == var_on){
                            for(j=0; j<dim_of_points; j++){
                                aggrigate[j] -= grad_mu[j];
                            }
                        }

                        for(j=0; j<dim_of_points; j++){
                            for(k=0; k<size_of_to_sample; k++){
                                aggrigate[j] -= grad_chol_decomp[winner][k][j]*normals[k]; // w.r.t the WINNER var
                            }
                        }
                    }
                }

                // write out
                for(j=0; j<dim_of_points; j++){
                    exp_grad_EI[idx*dim_of_points + j] = aggrigate[j]/(float)exp_grad_its;
                }

                // free memory
                free(aggrigate);
                free(EI_this_step_from_var);
                free(mu);
                free(grad_mu);
                free(normals);
                for(i=0; i<size_of_to_sample; i++){
                    free(cholesky_decomp[i]);
                    for(j=0; j<size_of_to_sample; j++){
                        free(grad_chol_decomp[i][j]);
                    }
                    free(grad_chol_decomp[i]);
                }
                free(cholesky_decomp);
                free(grad_chol_decomp);
            }
        }
        """),no_extern_c=True)

    get_grad_EI = mod.get_function("get_grad_EI")

    # GPP, points_to_sample=None, n_cores=2, total_restarts=100, grad_EI_its=1000, pr_steps=100


    assert(n_cores<=max_block_size)
    num_small_blocks = max_block_size/n_cores

    # other input
    dim_of_points = numpy.int32(len(GPP.points_sampled[0].point))
    best_so_far = numpy.float32(GPP.best_so_far)
    max_int_steps = numpy.int32(grad_EI_its)
    # pk params
    gamma = numpy.float32(gamma)
    pre_mult = numpy.float32(pre_mult)

    best_score_so_far = -1

    path_storage = {}

    for restart_on in range(total_restarts):

        # get starting points from a latin hypercube
        current_points = []
        for i in range(num_small_blocks):
            current_points.append(GPP_math.get_latin_hypercube_points(n_cores, GPP.domain))
            path_storage[(i, restart_on)] = []
        #print current_points

        for step_on in range(1, pr_steps+1):

            exp_grad_EI = numpy.zeros(n_cores*num_small_blocks*dim_of_points, dtype=numpy.float32)

            to_sample_mean_in = []
            chol_decomp_var_in = []
            grad_mu_in = []
            grad_chol_in = []

            for i, small_block_points in enumerate(current_points):

                path_storage[(i, restart_on)].append(small_block_points.copy())
                union_of_points = []
                union_of_points.extend(small_block_points)
                if points_being_sampled != None:
                    union_of_points.extend(points_being_sampled)
                size_of_to_sample = numpy.int32(len(union_of_points))

                to_sample_mean, cholesky_decomp_sample_var = GPP.get_mean_and_var_of_points(union_of_points)

                #print cholesky_decomp_sample_var

                to_sample_mean_in.extend(to_sample_mean)
                chol_decomp_var_in.extend(cholesky_decomp_sample_var.flatten())

                grad_mu_in.extend(GPP.get_grad_mu(union_of_points).flatten())

                #print "anyalytic grad ei"
                #for i in union_of_points:
                    #print GPP.get_expected_grad_EI(i, [])
                grad_K_star_star = GPP.build_grad_sample_covariance_matrix(union_of_points)
                grad_K_star = numpy.array(GPP.build_grad_K_star(union_of_points))
                K_star = numpy.array(GPP.build_mix_covariance_matrix(union_of_points))

                component = []
                for dim_on in range(dim_of_points):
                    component.append(numpy.dot(numpy.dot(grad_K_star[:,:,dim_on], GPP.K_inv), K_star.T))

                grad_chol_part = []
                for i in range(size_of_to_sample):
                    for j in range(size_of_to_sample):
                        vec2 = []
                        for dim_on in range(dim_of_points):
                            if i == j:
                                vec2.append(grad_K_star_star[i][j][dim_on] - 2*component[dim_on][i][j])
                            else:
                                vec2.append(grad_K_star_star[i][j][dim_on] - component[dim_on][i][j])
                        grad_chol_part.extend(vec2)

                grad_chol_in.extend(grad_chol_part)

            to_sample_mean_in = numpy.array(to_sample_mean_in, dtype=numpy.float32)
            chol_decomp_var_in = numpy.array(chol_decomp_var_in, dtype=numpy.float32)
            grad_mu_in = numpy.array(grad_mu_in, dtype=numpy.float32)
            grad_chol_in = numpy.array(grad_chol_in, dtype=numpy.float32)

            get_grad_EI(
                drv.InOut(exp_grad_EI),
                drv.In(to_sample_mean_in),
                drv.In(grad_mu_in),
                drv.In(chol_decomp_var_in),
                drv.In(grad_chol_in),
                dim_of_points, size_of_to_sample, best_so_far,
                max_int_steps, numpy.int32(n_cores),
                numpy.uint32((time.time()*1000000)),
                block=(int(num_small_blocks*n_cores),1,1), grid=(1,1))

            #print "Grad EI"
            #print exp_grad_EI

            # need to unflatten exp_grad_EI
            num_on = 0
            for block_on, small_block_points in enumerate(current_points):
                for point_on, point in enumerate(small_block_points):
                    for i in range(dim_of_points):
                        #print "adding %f to [%i,%i] %f->%f" % (pre_mult*pow(step_on, -gamma)*exp_grad_EI[num_on], block_on, point_on, point[i], point[i] + pre_mult*pow(step_on, -gamma)*exp_grad_EI[num_on])
                        point[i] += pre_mult*pow(step_on, -gamma)*exp_grad_EI[num_on]
                        num_on += 1
                    point = project_back_to_domain(point, GPP.domain)

            print "step (%i/%i) of (%i/%i)" % (step_on, pr_steps, restart_on+1, total_restarts)
            #print current_points

        print "Calculating best set of points so far..."

        for i, small_block_points in enumerate(current_points):
            path_storage[(i, restart_on)].append(small_block_points.copy())
            union_of_points = []
            union_of_points.extend(small_block_points)
            if points_being_sampled != None:
                union_of_points.extend(points_being_sampled)
            score_for_block = cuda_get_exp_EI(GPP, union_of_points)
            #print "block", small_block_points
            #print "EI: %f" % score_for_block

            if score_for_block > best_score_so_far:
                best_score_so_far = score_for_block
                best_block_so_far = small_block_points

    print "After %i attempts the best block was" % (total_restarts*n_cores)
    print best_block_so_far
    print "with score %f" % best_score_so_far

    if paths_only:
        return path_storage, best_block_so_far

    return best_block_so_far

def project_back_to_domain(point, domain):
    for dim, component in enumerate(point):
        if component > domain[dim][1]:
            point[dim] = domain[dim][1]
        elif component < domain[dim][0]:
            point[dim] = domain[dim][0]
    return point

def cuda_get_exp_EI(GPP, to_sample, iterations=1000000):
    mod = SourceModule("""
        #include <curand_kernel.h>

        extern "C"
        {
            __global__ void get_exp_EI(float *dest, float *cholesky_decomp, float *to_sample_mean, int sample_size, int NUM_ITS, float best_so_far, int rand_seed)
            {
                const int idx = threadIdx.x;
                int i,j,mc;

                // RNG setup
                unsigned int seed = rand_seed + idx*NUM_ITS;
                curandState s;
                // seed a random number generator
                curand_init(seed, 0, 0, &s);

                float *normals = (float *)malloc(sizeof(float)*sample_size);

                float agg = 0.0;
                float imp_this_step;
                float EI;

                int num_on;

                for(mc=0; mc<NUM_ITS; mc++){
                    imp_this_step = 0.0;
                    num_on = 0;

                    for(i=0; i<sample_size; i++){
                        normals[i] = curand_normal(&s);
                    }

                    for(i=0; i<sample_size; i++){
                        EI = best_so_far - to_sample_mean[i];
                        for(j=i; j<sample_size; j++){
                            EI -= cholesky_decomp[num_on]*normals[j];
                            num_on++;
                        }
                        if(EI > imp_this_step){
                            imp_this_step = EI;
                        }
                    }
                    agg += imp_this_step;
                }
                dest[idx] = agg/(float)NUM_ITS;

                free(normals);
            }
        }
    """,no_extern_c=True)

    best_so_far = numpy.float32(GPP.best_so_far)
    sample_size = len(to_sample)
    to_sample_mean, to_sample_var = GPP.get_mean_and_var_of_points(to_sample)
    cholesky_mat = GPP_math.cholesky_decomp(to_sample_var).T
    cholesky_decomp = numpy.zeros(((sample_size+1)*sample_size)/2)
    num_on = 0
    for i in range(sample_size):
        for j in range(sample_size):
            if j >= i:
                cholesky_decomp[num_on] = cholesky_mat[i][j]
                num_on += 1

    get_exp_EI = mod.get_function("get_exp_EI")

    cholesky_decomp = cholesky_decomp.astype(numpy.float32)
    to_sample_mean = to_sample_mean.astype(numpy.float32)

    agg = []

    for _ in range(numpy.max((iterations/100000, 1))):
        # each loop estimates EI over 100,000 iterations (250 per core, 400 cores)
        max_its = 250

        dest = numpy.zeros(400, dtype=numpy.float32)
        get_exp_EI(
            drv.Out(dest), drv.In(cholesky_decomp), drv.In(to_sample_mean),
            numpy.int32(sample_size), numpy.int32(max_its), numpy.float32(best_so_far), numpy.int32((time.time()*1000000)),
            block=(400,1,1), grid=(1,1))

        agg.append(numpy.mean(dest))

    return numpy.mean(agg)

def main():
    GPP = GP.GaussianProcess()

    GPP.domain = [[-1,1],[-1,1]]

    #for point in GPP_math.get_latin_hypercube_points(1, GPP.domain):
    for point in numpy.arange(-1,2,0.1):
        s = GP.SamplePoint(numpy.array([point, point]), point, 0)
        GPP.add_sample_point(s)

    #GPP_plotter.plot_1D_EI(GPP, domain=numpy.arange(-1,1,0.01))
    #GPP_plotter.plot_GPP(GPP, domain=numpy.arange(-1,1,0.01))

    #current_point = GPP_math.get_latin_hypercube_points(num_points, [[0,20],[0,20]])

    #t = time.time()
    #print cuda_get_exp_EI(GPP, current_point, iterations=1000000)
    #print num_points, "TIME: ", time.time()-t

    #return 0

    being_sampled = []
    #being_sampled = GPP_math.get_latin_hypercube_points(1, GPP.domain)

    t = time.time()
    cuda_get_next_points_new(GPP, n_cores=4, max_block_size=12, grad_EI_its=1000, total_restarts=1, pr_steps=5, points_being_sampled=being_sampled)
    print "TIME: ", time.time()-t

    #t = time.time()
    #print GPP.get_expected_improvement(to_sample, iterations=40000)
    #print time.time()-t
    #t = time.time()
    #print cuda_get_exp_EI(GPP, to_sample)
    #print time.time()-t

if __name__ == '__main__':
    main()
