# -*- coding: utf-8 -*-
"""Views for the MOE frontend and REST interface.

Contains:

    * :mod:`moe.views.frontend`: the frontend code
    * :mod:`moe.views.rest`: various REST endpoints for internal gaussian process information
    * :mod:`moe.views.pretty_view`: base view for all REST endpoints
    * :mod:`moe.views.bandit_pretty_view`: base view for all bandit REST endpoints
    * :mod:`moe.views.gp_pretty_view`: base view for all GP REST endpoints
    * :mod:`moe.views.optimizable_gp_pretty_view`: base view for REST endpoints that require optimization
    * :mod:`moe.views.gp_next_points_pretty_view`: base view for getting the next best points to sample
    * :mod:`moe.views.schemas`: schemas used to deserialize/serialize inputs/outputs in the REST interface
    * :mod:`moe.views.utils`: utils for constructing data structures/classes from :mod:`moe.optimal_learning.python`
    * :mod:`moe.views.constant`: constants shared by multiple views
    * :mod:`moe.views.exceptions`: exception handling views for giving detailed 500 errors

"""
