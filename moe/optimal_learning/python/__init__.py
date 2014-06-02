# -*- coding: utf-8 -*-
"""The python component of the optimal_learning package, containing wrappers around C++ implementations of features and Python implementations of some of those features.

**Files in this package**

* constant.py: some default configuration values for ``optimal_learning`` components
* data_containers.py: SamplePoint and HistoricalData containers for passing data to the ``optimal_learning`` library
* geometry_utils.py: geometry utilities; e.g., ClosedInterval, random point generation
* linkers.py: linkers connecting equivalent ``cpp_wrappers`` and ``python_version`` versions of ``optimal_learning`` components.
* repeated_domain.py: RepeatedDomain object for manipulating sets of points simultaneously within the same domain

**Major sub-packages**

**interfaces**

A set of abstract base classes (ABCs) defining an interface for interacting with optimal_learning. These consist of composable
functions and classes to build models, perform model selection, and design new experiments.

**cpp_wrappers**

An implementation of the ABCs in interfaces using wrappers around (fast) C++ calls. These routines are meant for "production" runs
where high performance is a concern.

.. Note:: the higher level C++ interfaces are generally *not* composable with objects not in the cpp_wrappers package. So it
  would be possible to implement ExpectedImprovementInterface in Python and connect it to
  cpp_wrappers.gaussian_process.GaussianProcess, BUT it is not currently possible to connect
  cpp_wrappers.expected_improvement.ExpectedImprovement to python_version.gaussian_process.GaussianProcess.

**python_version**

An implementation of the ABCs in interfaces using Python (with numpy/scipy). These routines are more for educational and
experimental purposes. Python is generally simpler than C++ so the hope is that this package is more accessible to new
users hoping to learn about optimal_learning. Additionally, development time in Python is shorter, so it could be convenient
to test new ideas here before fully implementing them in C++. For example, developers could test a new OptimizationInterface
implementation in Python while connecting it to C++ evaluation of objective functions.

"""
