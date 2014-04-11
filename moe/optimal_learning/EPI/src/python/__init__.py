# -*- coding: utf-8 -*-
"""The python component of the optimal_learning package, containing wrappers around C++ implementations of features and Python implementations of some of those features.

Major sub-packages:
interfaces:
A set of abstract base classes (ABCs) defining an interface for interacting with optimal_learning. These consist of composable
functions and classes to build models, perform model selection, and design new experiments.

cpp_wrappers:
An implementation of the ABCs in interfaces using wrappers around (fast) C++ calls.

models:
A deprecated package containing the old C++ and Python implementations.

"""
