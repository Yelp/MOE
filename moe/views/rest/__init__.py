# -*- coding: utf-8 -*-
"""The REST interface for the MOE webapp.

**Internal Gaussian Process (GP) endpoints:**

    * :mod:`~moe.views.rest.gp_ei`

        .. http:post:: /gp/ei

           Calculates the Expected Improvement (EI) of a set of points, given historical data.

        .. http:get:: /gp/ei/pretty

    * :mod:`~moe.views.rest.gp_mean_var`

        .. http:post:: /gp/mean_var

           Calculates the GP mean and covariance of a set of points, given historical data.

        .. http:get:: /gp/mean_var/pretty

**Next points endpoints:**

    * :mod:`~moe.views.rest.gp_next_points_epi`

        .. http:post:: /gp/next_points/epi

           Calculates the next best points to sample, given historical data, using Expected Parallel Improvement (EPI).

        .. http:get:: /gp/next_points/epi/pretty

    * :mod:`~moe.views.rest.gp_next_points_constant_liar`

        .. http:post:: /gp/next_points/constant_liar

           Calculates the next best points to sample, given historical data, using Constant Liar (CL).

        .. http:get:: /gp/next_points/constant_liar/pretty

    * :mod:`~moe.views.rest.gp_next_points_kriging`

        .. http:post:: /gp/next_points/kriging

           Calculates the next best points to sample, given historical data, using Kriging.

        .. http:get:: /gp/next_points/kriging/pretty

**Bandit endpoints:**

    * :mod:`~moe.views.rest.bandit_epsilon`

        .. http:post:: /bandit/epsilon

           Calculates the arm allocations and the best arm to pull next using Epsilon policy, given subtype, historical data, hyperparameters.

        .. http:get:: /bandit/epsilon/pretty

    * :mod:`~moe.views.rest.bandit_ucb`

        .. http:post:: /bandit/ucb

           Calculates the arm allocations and the best arm to pull next using UCB policy, given subtype, historical data, hyperparameters.

        .. http:get:: /bandit/ucb/pretty

    * :mod:`~moe.views.rest.bandit_bla`

        .. http:post:: /bandit/bla

           Calculates the arm allocations and the best arm to pull next using BLA policy, given subtype, historical data, hyperparameters.

        .. http:get:: /bandit/bla/pretty

"""
