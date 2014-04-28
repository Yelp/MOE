# -*- coding: utf-8 -*-
"""The python component of the optimal_learning package, containing wrappers around C++ implementations of features and Python implementations of some of those features.

Major sub-packages:
* interfaces:
A set of abstract base classes (ABCs) defining an interface for interacting with optimal_learning. These consist of composable
functions and classes to build models, perform model selection, and design new experiments.

* cpp_wrappers:
An implementation of the ABCs in interfaces using wrappers around (fast) C++ calls. These routines are meant for "production" runs
where high performance is a concern.

.. Note:: the higher level C++ interfaces are generally *not* composable with objects not in the cpp_wrappers package. So it
  would be possible implement ExpectedImprovementInterface in Python and connect it to cpp_wrappers.gaussian_process.GaussianProcess,
  BUT it is not currently possible to connect cpp_wrappers.expected_improvement.ExpectedImprovement to
  python_version.gaussian_process.GaussianProcess.

* python_version:
An implementation of the ABCs in interfaces using Python (with numpy/scipy). These routines are more for educational and
experimental purposes. Python is generally simpler than C++ so the hope is that this package is more accessible to new
users hoping to learn about optimal_learning. Additionally, development time in Python is shorters, so it could be convenient
to test new ideas here before fully implementing them in C++. For example, developers could test a new OptimizationInterface
implementation in Python while connecting it to C++ evaluation of objective functions.

.. Note:: Not implemented yet: ADS-3789

* models:
A deprecated package containing the old C++ and Python implementations. This package (and associated tests) will be deleted
after python_version is implemented: ADS-3987.

"""
