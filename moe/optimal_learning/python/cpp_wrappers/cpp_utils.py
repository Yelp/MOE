# -*- coding: utf-8 -*-
"""Utilities for making data C++ consumable and for making C++ outputs Python consumable."""
import numpy


def cppify(array):
    """Flatten a numpy array and copies it to a list for C++ consumption.

    TODO(GH-159): This function will be unnecessary when C++ accepts numpy arrays.

    :param array: array to convert
    :type array: array-like (e.g., ndarray, list, etc.) of float64
    :return: list copied from flattened array
    :rtype: list

    """
    return list(numpy.ravel(array))


def uncppify(array, expected_shape):
    """Reshape a copy of the input array into the expected shape.

    TODO(GH-159): If C++ returns numpy arrays, we can kill this function (instead, call reshape directly).

    :param array: array to reshape
    :type array: array-like
    :param expected_shape: desired shape for array
    :type expected_shape: int or tuple of ints
    :return: reshaped input
    :rtype: array of float64 with shape ``expected_shape``

    """
    return numpy.reshape(array, expected_shape)


def cppify_hyperparameters(hyperparameters):
    r"""Convert a flat array of hyperparameters into a form C++ can consume.

    C++ interface expects hyperparameters in a list, where:
    hyperparameters[0]: ``float64 = \alpha`` (``\sigma_f^2``, signal variance)
    hyperparameters[1]: list = length scales (len = dim, one length per spatial dimension)

    :param hyperparameters: hyperparameters to convert
    :type hyperparameters: array of float64 with shape (num_hyperparameters)
    :return: hyperparameters converted to C++ input format
    :rtype: list where item [0] is a float and item [1] is a list of float with len = dim

    """
    return [numpy.float64(hyperparameters[0]), cppify(hyperparameters[1:])]
