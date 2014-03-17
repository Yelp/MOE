#!/usr/bin/python

# Copyright (C) 2012 Scott Clark. All rights reserved.

#import ipdb # debugger

import logging # logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
#from optparse import OptionParser # parser = OptionParser()

#import numpy # for sci comp
#import matplotlib.pylab as plt # for plotting

#import time # for timing
#import commands # print commands.getoutput(script)

import src.python.models.optimal_gaussian_process
import src.python.models.sample_point

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def _setup_GP():
    # Make a gaussian process
    GP = src.python.models.optimal_gaussian_process.OptimalGaussianProcess(domain=[[0,2],[0,2]])

    # Add some points
    sample_point = src.python.models.sample_point.SamplePoint([0,0], 1)
    GP.add_sample_point(sample_point)
    sample_point = src.python.models.sample_point.SamplePoint([1,0], 2)
    GP.add_sample_point(sample_point)
    return GP

def test_making_GP_and_adding_points():
    GP = _setup_GP()
    logging.info("test_making_GP_and_adding_points PASS")

def test_optimal_GP_find_next_point(num_points=4):
    GP = _setup_GP()
    starting_points = src.python.lib.math.get_latin_hypercube_points(num_points, GP.domain)
    next_point = GP.get_next_step_multistart(starting_points, [])
    logging.info("test_optimal_GP_find_next_point PASS")

def main():
    test_making_GP_and_adding_points()
    test_optimal_GP_find_next_point()

if __name__ == '__main__':
    main()
