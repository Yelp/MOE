# -*- coding: utf-8 -*-
"""Bandit directory containing multi-armed bandit implementation in python.

**Files in this package**

* :mod:`moe.bandit.constant`: some default configuration values for ``optimal_learning`` components
* :mod:`moe.bandit.data_containers`: :class:`~moe.bandit.data_containers.SampleArm`
  and :class:`~moe.bandit.data_containers.HistoricalData` containers for passing data to the ``bandit`` library
* :mod:`moe.bandit.linkers`: linkers connecting ``bandit`` components.

**Interfaces**
:mod:`moe.bandit.bandit_interface`

**Bandit packages**
:mod:`moe.bandit.epsilon`: Epsilon bandit policies
:mod:`moe.bandit.ucb`: UCB bandit policies
:mod:`moe.bandit.bla`: BLA bandit policies

A set of abstract base classes (ABCs) defining an interface for interacting with ``bandit``. These consist of composable
functions and classes to allocate bandit arms and choose arm.

"""
