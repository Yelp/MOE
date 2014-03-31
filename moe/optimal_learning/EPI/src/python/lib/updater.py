#!/usr/bin/python

# (C) 2011 Scott Clark

import numpy # for sci comp
#import matplotlib.pylab as plt # for plotting
#import time # for timing
#import commands # print commands.getoutput(script)

import GaussianProcessPrior as gp
import GPP_math

def get_next_step(GPP, starting_point, points_being_sampled, domain=None, gamma=0.8, iterations=1000, max_N=400):
    """get the next step using polyak-ruppert"""
    x_hat = starting_point
    x_hat_prev = starting_point*2

    x_path = [starting_point]

    n = 1
    while n < max_N:
        alpha_n = 0.01*n**-gamma

        gx_n = GPP.get_expected_grad_EI(x_path[-1], points_being_sampled, iterations=10)

        x_path.append(x_path[-1] + alpha_n*gx_n)

        if n%(max_N/20) == 0:
            x_hat_prev = x_hat
            x_hat = numpy.mean(x_path[1:], axis=0)

        n += 1

    union_of_points = [x_hat]
    union_of_points.extend(points_being_sampled)
    EI = GPP.get_expected_improvement(union_of_points)

    return x_path, x_hat, EI

def get_next_step_multistart(GPP, starting_points, points_being_sampled, domain=None, gamma=0.9, iterations=1000, max_N=100):
    """get next step using multistart polyak-ruppert"""
    polyak_ruppert_paths = []
    best_improvement = -numpy.inf
    best_point = None
    for starting_point in starting_points:
        x_path, x_hat, EI = get_next_step(GPP, starting_point, points_being_sampled, domain, gamma=gamma, iterations=iterations, max_N=max_N)
        if not domain or not GPP_math.not_in_domain(x_hat, domain):
            polyak_ruppert_paths.append([x_path, x_hat, EI])
            if polyak_ruppert_paths[-1][2] > best_improvement:
                best_improvement = polyak_ruppert_paths[-1][2]
                best_point = polyak_ruppert_paths[-1][1]

    return best_point, best_improvement, polyak_ruppert_paths

def get_next_step_fmin(GPP, starting_point, points_being_sampled, domain=None):
    """get next step using deriv free simplex method"""
    def func_to_min(x):
        if domain and GPP_math.not_in_domain(x, domain):
            return numpy.inf
        union_of_points = [x]
        if points_being_sampled:
            union_of_points.extend(points_being_sampled)
            return -(GPP.get_expected_improvement(union_of_points) - GPP.get_expected_improvement(points_being_sampled))
        else:
            return -(GPP.get_expected_improvement(union_of_points))

    fmin_out = fmin(func_to_min, starting_point, xtol=1e-3, full_output = 1, disp=0)
    x_hat = fmin_out[0]
    EI = -fmin_out[1]

    return x_hat, EI

def get_fmin_multistart(GPP, multistart_points, points_being_sampled, domain=None):
    """get the best next step given a set of multistart points (and maybe a domain)"""
    best_seen = -numpy.inf
    best_point = None
    for point in multistart_points:
        xopt, EI = get_next_step_fmin(GPP, point, points_being_sampled)
        if domain:
            if GPP_math.not_in_domain(xopt, domain) and EI > best_seen:
                best_point = xopt[0]
        else:
            if EI > best_seen:
                best_point = xopt[0]

    return best_point

def get_multistart_best_fmin(GPP, domain, random_restarts=5, points_being_sampled=[]):
    best_improvement = -numpy.inf
    #random_restarts = len(GPP.points_sampled)*4
    for _ in range(random_restarts):
        next_step, improvement = get_next_step_fmin(GPP, GPP_math.make_rand_point(domain), points_being_sampled, domain=domain)
        #path, next_step, improvement = GPP.get_next_step(numpy.array([-2.141592, 11.274999]), [numpy.array([1.0,0.0])], [0,1])
        if improvement > best_improvement:
            best_improvement = improvement
            best_next_step = next_step
    return best_next_step

def get_multistart_best(GPP, domain, random_restarts=5, points_being_sampled=[]):
    best_improvement = -numpy.inf
    best_next_step = GPP_math.make_rand_point(domain)
    #random_restarts = len(GPP.points_sampled)*4
    for _ in range(random_restarts):
        path, next_step, improvement = get_next_step(GPP, GPP_math.make_rand_point(domain), points_being_sampled, domain)
        #path, next_step, improvement = GPP.get_next_step(numpy.array([-2.141592, 11.274999]), [numpy.array([1.0,0.0])], [0,1])
        if improvement > best_improvement and not GPP_math.not_in_domain(next_step, domain):
            best_improvement = improvement
            best_next_step = next_step
    return best_next_step

def main():
    pass

if __name__ == '__main__':
    main()
