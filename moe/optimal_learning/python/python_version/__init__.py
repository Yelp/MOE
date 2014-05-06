# -*- coding: utf-8 -*-
"""Implementations of the ABCs in the python/interfaces package using Python (as opposed to C++ calls).

The modules in this package are meant to fill two main purposes:
1) Provide a (hopefully) easy-to-read Python implementation to help familiarize people (especially those who are not
   well-versed in C++) with the features of the optimal_learning library.
2) Provide a convenient work/test environment for developers to try out new algorithms, features, and ideas. If you have
   a new way to compute ExpectedImprovement, you can quickly develop the algorithm in Python and then use either
   python_version/gaussian_process.py or (faster) cpp_wrappers/gaussian_process.py. Or you can test out some new
   optimization methods in Python and connect your optimizers to the objective functions in this package or
   offload the expensive computation to C++ via cpp_wrappers.

Unlike the interface implementations in the cpp_wrappers package, the classes in this package are all composable with
each other and with classes in cpp_wrappers.

Modules in this package make extensive use of numpy and scipy. As mentioned in the interface documentation, all functions
return and accept numpy arrays.

See the package comments for python/interfaces (in __init__.py) for an overview of optimal_learning's capabilities.

"""
