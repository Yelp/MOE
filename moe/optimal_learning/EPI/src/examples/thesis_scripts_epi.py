# -*- coding: utf-8 -*-
#!/usr/bin/python

# Copyright (C) 2012 Scott Clark. All rights reserved.

#import pdb # debugger

#import logging # logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
#from optparse import OptionParser # parser = OptionParser()

import numpy # for sci comp
import matplotlib.pylab as plt # for plotting

import time # for timing
#import commands # print commands.getoutput(script)

import GaussianProcessPrior as GP
import GPP_math
import GPP_cuda
import GPP_plotter

def branin_func(input_vec):
    x1 = input_vec[0]
    x2 = input_vec[1]
    # enforce boundary conditions
    if x1 > 10 or x1 < -5 or x2 < 0 or x2 > 15:
        raise(ValueError, "Outside of bounds for Branin function")
    return (x2 - (5.1/(4*numpy.pi**2))*x1**2 + (5/numpy.pi)*x1 - 6)**2 + 10*(1 - 1.0/(8*numpy.pi))*numpy.cos(x1) + 10

def plot_2D_GPP(GPP, save_figure=False, pdf_stream=None, passed_fig=None, fig_place=122, EI_on=False):

    num_points = float(15)
    # set up mesh
    x1_delta = (GPP.domain[0][1] - GPP.domain[0][0])/num_points
    domain_x1 = numpy.arange(GPP.domain[0][0], GPP.domain[0][1], x1_delta)
    x2_delta = (GPP.domain[1][1] - GPP.domain[1][0])/num_points
    domain_x2 = numpy.arange(GPP.domain[1][0], GPP.domain[1][1], x2_delta)
    X, Y = numpy.meshgrid(domain_x1, domain_x2)

    sample_mean = numpy.zeros((len(domain_x1), len(domain_x2)))
    #exp_EI = numpy.zeros((len(domain_x1), len(domain_x2)))

    # find sample_mean
    for i, x1 in enumerate(domain_x1):
        for j, x2 in enumerate(domain_x2):
            if not EI_on:
                mu, var = GPP.get_mean_and_var_of_points([[x1,x2]])
                #print x1,x2,mu
                sample_mean[j][i] = mu
            else:
                EI = GPP_cuda.cuda_get_exp_EI(GPP, [[x1,x2]], iterations=1000)
                #print x1,x2,EI
                sample_mean[j][i] = EI

    # create figure
    if not passed_fig:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = passed_fig
        ax = fig.add_subplot(fig_place)

    for sampled in GPP.points_sampled:
        ax.plot([sampled.point[0]], [sampled.point[1]], 'rx')

    CS = plt.contour(X, Y, sample_mean)

    ax.set_xlim(numpy.min(domain_x1)-x1_delta, numpy.max(domain_x1)+x1_delta)
    ax.set_ylim(numpy.min(domain_x2)-x2_delta, numpy.max(domain_x2)+x2_delta)

    # save or show figure
    if save_figure:
        if not pdf_stream:
            return fig, ax
        else:
            pdf_stream.savefig()
    else:
        plt.show()

def plot_polyak_ruppert_paths(steps_to_take=10):
    covariance_of_process = GP.CovarianceOfProcess(l=5.0, a=10.0)
    GPP = GP.GaussianProcess(covariance_of_process=covariance_of_process)
    GPP.domain = [[-5,10],[0,15]]

    starting_stencil = []
    for x in numpy.arange(-5,15,7.5):
        for y in numpy.arange(0,20,7.5):
            s = GP.SamplePoint(numpy.array([x,y]), branin_func([x,y]), 0)
            GPP.add_sample_point(s)

    for _ in range(steps_to_take):

        fig, ax = plot_2D_GPP(GPP, save_figure=True, EI_on=True)
        print "contour generated"
        paths, best_block = GPP_cuda.cuda_get_next_points_new(GPP, n_cores=1, max_block_size=64, grad_EI_its=1000, total_restarts=4, pr_steps=10, points_being_sampled=[], paths_only=True, pre_mult=1)

        #print "paths", paths

        color_rotation = ['b','m','g','r','c','k']
        color_size = len(color_rotation)
        color_on = 0

        for path_key in paths:
            path = paths[path_key]
            #print "path", path
            path_x = []
            path_y = []
            for _ in range(len(path[0])):
                path_x.append([])
                path_y.append([])

            #print path_x
            for set_on, set_of_points in enumerate(path):
                #print path_x
                #print "set_of_points", set_of_points
                for point_on, point in enumerate(set_of_points):
                    #print "point", point_on, point
                    path_x[point_on].append(point[0])
                    path_y[point_on].append(point[1])
                    #print path_x
            for i, xs in enumerate(path_x):
                #print xs, path_y[i]
                ax.plot(xs, path_y[i], color_rotation[color_on%color_size] + 'x-')
                ax.plot([xs[-1]], [path_y[i][-1]], color_rotation[color_on%color_size] + '*') # ending point
            color_on += 1

        ax.set_xlim(numpy.min(GPP.domain[0]), numpy.max(GPP.domain[0]))
        ax.set_ylim(numpy.min(GPP.domain[1]), numpy.max(GPP.domain[1]))

        s = GP.SamplePoint(numpy.array(best_block[0]), branin_func(best_block[0]), 0)
        print s
        GPP.add_sample_point(s)

def core_speed_two():
    number_of_runs = 20
    wall_time = 10
    num_cores = [1,2,4,8]

    domain = [[-16,16]]
    stencil = [-16,0,16]

    default_hyper = [1.0,1.0,0.1]

    for run_on in range(number_of_runs):
        t_run_start = time.time()
        fout = open("speed_test_r%i.txt" % (run_on), "w")

        GPP_true_function = GP.GaussianProcess(covariance_of_process=GP.CovarianceOfProcess(hyperparameters=default_hyper))
        GPP_true_function.domain = domain

        stencil_vals = []
        for point in stencil:
            s = GP.SamplePoint(numpy.array([point]), GPP_true_function.sample_from_GP(numpy.array([point])))
            stencil_vals.append(s.value)
            GPP_true_function.add_sample_point(s)


        for n_cores in num_cores:
            t_core_start = time.time()
            gain = [0]

            GPP = GP.GaussianProcess(covariance_of_process=GP.CovarianceOfProcess(hyperparameters=default_hyper))
            GPP.domain = domain

            for i, point in enumerate(stencil):
                s = GP.SamplePoint(numpy.array([point]), stencil_vals[i])
                GPP.add_sample_point(s)

            points_being_sampled = []
            for _ in range(n_cores):
                new_point_to_sample = GPP_cuda.cuda_get_next_points_new(GPP, n_cores=1, max_block_size=64, total_restarts=1, points_being_sampled=points_being_sampled, pr_steps=10)
                points_being_sampled.append(numpy.array(new_point_to_sample))


            for sim_on in range(n_cores*wall_time):
                t_sim_start = time.time()

                pre_best = GPP.best_so_far
                # pull off oldest points_being_sampled and sample it
                s = GP.SamplePoint(points_being_sampled[0], GPP_true_function.sample_from_GP(points_being_sampled[0]))
                GPP_true_function.add_sample_point(s)
                GPP.add_sample_point(s)

                points_being_sampled.pop(0)
                # find next point to sample
                if sim_on < n_cores*(wall_time-1) + 1:
                    points_being_sampled.append(GPP_cuda.cuda_get_next_points_new(GPP, n_cores=1, max_block_size=64, total_restarts=1, points_being_sampled=points_being_sampled, pr_steps=10))

                if pre_best > GPP.best_so_far:
                    gain.append(gain[-1] + (pre_best - GPP.best_so_far))
                else:
                    gain.append(gain[-1])

                print "Finished sim_on=%i (of %i) (n_cores=%i) in %f" % (sim_on, n_cores*wall_time, n_cores, time.time() - t_sim_start)

            print "Finished n_cores=%i in %f" % (n_cores, time.time()-t_core_start)

            fout.write("cores:%i\n" % (n_cores))
            tot = 0
            for g_on, g in enumerate(gain[1:]):
                tot += g
                if (g_on+1)%n_cores == 0:
                    fout.write("%f," % (tot/float(n_cores)))
                    tot = 0
            fout.write("\n")
        print "Finished run_on=%i (of %i) in %f" % (run_on+1, number_of_runs, time.time()-t_run_start)
        fout.close()



def core_speed_tests_block():
    number_of_runs = 10
    wall_time = 24
    num_cores = [1,2,4,6,8,12,24]

    fout = open("core_speed_tests.txt", "w")

    gain_dict = {}
    for n_cores in num_cores:
        gain_dict[n_cores] = numpy.zeros(wall_time+1)

    sample_dict = {}
    for n_cores in num_cores:
        sample_dict[n_cores] = []

    value_dict = {}
    for n_cores in num_cores:
        value_dict[n_cores] = []

    time_dict = {}
    for n_cores in num_cores:
        time_dict[n_cores] = numpy.zeros(wall_time+1)

    for run_on in range(number_of_runs):

        GPP_true_function = GP.GaussianProcess(covariance_of_process=GP.CovarianceOfProcess(hyperparameters=[1.0,1.0,0.1]))
        GPP_true_function.domain = [[-8,8]]

        for point in [-8, 0, 8]:
            s = GP.SamplePoint(numpy.array([point]), GPP_true_function.sample_from_GP(numpy.array([point])))
            GPP_true_function.add_sample_point(s)

        for n_cores in num_cores:
            gain = [0]
            sampled = []
            values = []
            t0 = time.time()
            GPP = GP.GaussianProcess(covariance_of_process=GP.CovarianceOfProcess(hyperparameters=[1.0,1.0,0.1]))
            GPP.domain = [[-8,8]]

            for point in GPP_true_function.points_sampled:
                GPP.add_sample_point(point)
                sampled.append(point.point[0])
                values.append(point.value)

            for t in range(wall_time):
                points = GPP_cuda.cuda_get_next_points_new(GPP, n_cores=n_cores, max_block_size=64, total_restarts=n_cores, pr_steps=10)
                pre_best = GPP.best_so_far
                for point in points:
                    sampled.append(point[0])
                    s = GP.SamplePoint(numpy.array([point]), GPP_true_function.sample_from_GP(numpy.array([point])))
                    GPP_true_function.add_sample_point(s)
                    GPP.add_sample_point(s)
                    values.append(s.value)
                if pre_best > GPP.best_so_far:
                    gain.append(gain[-1] + (pre_best - GPP.best_so_far))
                else:
                    gain.append(gain[-1])
            print gain
            print numpy.shape(gain)

            gain_dict[n_cores] += numpy.array(gain)
            time_dict[n_cores] += time.time() - t0
            print "time", time.time() - t0
            sample_dict[n_cores].append(sampled)
            value_dict[n_cores].append(values)
            print "finished %i %i/%i in %f" % (n_cores, run_on, number_of_runs, time.time() - t0)
        print gain_dict
        print sample_dict
        for n_cores in num_cores:
            fout.write("n_cores %i\n" % n_cores)
            for val in gain_dict[n_cores]:
                fout.write("%f " % val)
            fout.write("\nsampled\n")
            for i,sample in enumerate(sample_dict[n_cores][run_on]):
                fout.write("%f:%f " % (sample, value_dict[n_cores][run_on][i]))
            fout.write("\n")
    for n_cores in num_cores:
        gain_dict[n_cores] = gain_dict[n_cores]/float(number_of_runs)
        time_dict[n_cores] = time_dict[n_cores]/float(number_of_runs)
    print gain_dict, time_dict, sample_dict
    fout.close()
    return gain_dict, time_dict, sample_dict

def hyper_parameter_update_figure():

    cop = GP.CovarianceOfProcess(hyperparameters=[1.0,1.0,0.0])
    GPP_hidden = GP.GaussianProcess(covariance_of_process=cop)

    GPP = GP.GaussianProcess(covariance_of_process=GP.CovarianceOfProcess(hyperparameters=[1.0,1.0,0.0]))

    GPP.domain = [[-7,7]]

    for i, point in enumerate(GPP_math.get_latin_hypercube_points(20, GPP.domain)):
        s = GP.SamplePoint(numpy.array([point]), GPP_hidden.sample_from_GP(numpy.array([point])))
        GPP.add_sample_point(s)
        GPP_hidden.add_sample_point(s)

    GPP_plotter.plot_GPP(GPP, numpy.arange(-7,7,0.1))
    GPP_plotter.plot_GPP(GPP_hidden, numpy.arange(-7,7,0.1))

    GPP.update_cop()

    GPP_plotter.plot_GPP(GPP, numpy.arange(-7,7,0.1))

    return 0

    GPP_plotter.plot_GPP(GPP, numpy.arange(0,1,0.02))

    for i, point in enumerate(numpy.arange(0.1,1.2,0.2)):
        s = GP.SamplePoint(numpy.array([point]), GPP_hidden.sample_from_GP(numpy.array([point])))
        GPP.add_sample_point(s)
        GPP_hidden.add_sample_point(s)
    GPP.update_cop()

    GPP_plotter.plot_GPP(GPP, numpy.arange(0,1,0.02))

def exp_EI_speedup_iterations():
    num_points = 4
    restarts = 1
    GPU_speed = {100000:[], 200000:[], 500000:[], 1000000:[], 2000000:[], 5000000:[], 10000000:[], 20000000:[], 50000000:[], 100000000:[]}
    CPU_speed = {10000:[], 20000:[], 50000:[], 100000:[], 200000:[], 500000:[]}
    for _ in range(restarts):
        GPP = GP.GaussianProcess()

        GPP.domain = [[-1,1],[-1,1]]

        for point in numpy.arange(-1,2,1):
            s = GP.SamplePoint(numpy.array([point, point]), 0, 0)
            GPP.add_sample_point(s)

        to_sample = GPP_math.get_latin_hypercube_points(num_points, GPP.domain)

        for cpu_it in CPU_speed:
            t = time.time()
            EI = GPP.get_expected_improvement(to_sample, iterations=cpu_it)
            time_taken = time.time()-t
            print "CPU %i its: %f (ms), EI=%f" % (cpu_it, time_taken, EI)
            CPU_speed[cpu_it].append(time_taken)

        for gpu_it in GPU_speed:
            t = time.time()
            EI = GPP_cuda.cuda_get_exp_EI(GPP, to_sample, iterations=gpu_it)
            time_taken = time.time()-t
            print "GPU %i its: %f (ms), EI=%f" % (gpu_it, time_taken, EI)
            GPU_speed[gpu_it].append(time_taken)

    CPU_x = CPU_speed.keys()
    CPU_x.sort()
    CPU_y = []
    CPU_y_std = []
    for x in CPU_x:
        CPU_y.append(numpy.mean(CPU_speed[x]))
        CPU_y_std.append(numpy.std(CPU_speed[x]))

    GPU_x = GPU_speed.keys()
    GPU_x.sort()
    GPU_y = []
    GPU_y_std = []
    for x in GPU_x:
        GPU_y.append(numpy.mean(GPU_speed[x]))
        GPU_y_std.append(numpy.std(GPU_speed[x]))

    print CPU_x, CPU_y
    print GPU_x, GPU_y

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(CPU_x, CPU_y)
    ax.plot(GPU_x, GPU_y, 'x-')
    ax.set_xscale('log')
    ax.set_xlabel('number of iterations')
    ax.set_ylabel('wall clock time (s)')

def main():
    core_speed_two()

if __name__ == '__main__':
    main()
