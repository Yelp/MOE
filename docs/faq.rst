Frequently Asked Questions
**************************

Questions:

#. `What license is MOE released under?`_
#. `When should I use MOE?`_
#. `What is the time complexity of MOE?`_
#. `How do I cite MOE?`_
#. `Why does MOE take so long to return the next points to sample for some inputs?`_
#. `How do I bootstrap MOE? What initial data does it need?`_
#. `How many function evaluations do I need before MOE is "done"?`_
#. `How many function evaluations do I perform before I update the hyperparameters of the GP?`_
#. `Will you accept my pull request?`_

What license is MOE released under?
-----------------------------------

MOE is licensed under the `Apache License, Version 2.0`_

.. _Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0

When should I use MOE?
----------------------

MOE is designed for optimizing a system's parameters, when evaluating parameters is *time-consuming* or *expensive*, the objective function is a *black box* and not necessarily concave or convex, derivatives are unavailable, and we wish to find a global optimum, rather than a local one.

See :doc:`why_moe` for more information and links to other methods that solve similar problems.

What is the time complexity of MOE?
-----------------------------------

The most expensive part of MOE is when we form the Cholesky Decomposition of the covariance matrix. This operation is :math:`O(N^{3})` where *N* is the number of historical points. Within MOE techniques like Stochastic Gradient Descent and Monte Carlo integration are used, which are linear in the number of multistarts and iterations, these are tunable parameters.

For more information about the runtime of MOE check out :doc:`gpp_linear_algebra` and :doc:`gpp_math`.

How do I cite MOE?
------------------

For now just cite the repo: http://github.com/Yelp/MOE, a journal article with more technical detail should be coming out shortly as well.

Why does MOE take so long to return the next points to sample for some inputs?
------------------------------------------------------------------------------

Some optimizations are very computationally complex, or cause internal data structures like the covariance matrix between all historical points to become very large or ill formed (we need it to symmetric positive definite to machine precision).

A (non exhaustive) list of queries that are "hard" for MOE is:

 * Asking MOE for N new points to sample given H historical points where :math:`N >> H`. See `How do I bootstrap MOE? What initial data does it need?`_.
 * Asking MOE for new points to sample given H historical points where H is very large (thousands). See `When should I use MOE?`_, if you have several thousand historical points your problem may not be *time-consuming* or *expensive*. MOE performs best when every function evaluation is very difficult and we want to sample as few times as possible.
 * Asking MOE for many new points to sample using the :mod:`moe.views.rest.gp_next_points_kriging` method with no noise. The way this method works internally causes the covariance matrix to quickly grow in condition number, which causes many problems (like with the cholesky decomposition required for Stochastic Gradient Descent). Try using the :mod:`moe.views.rests.gp_next_points_epi` or :mod:`moe.views.rest.gp_next_points_constant_liar` endpoint, or allowing noise.

Check out `What is the time complexity of MOE?`_ for more information on timings.

How do I bootstrap MOE? What initial data does it need?
-------------------------------------------------------

MOE performs best when it has some initial, historical information to work with. Without any information it treats every point as equal Expected Improvement and will effectively choose points to sample at random (which is the best you can do with no information).

To help "bootstrap" MOE try:

 * Giving MOE all historical information possible, even if it has high noise. This can include previous experiments or the current status quo in an A/B test.
 * Try sampling a small `stencil`_ of points in the space you want MOE to search over. This is usually better than a random set of initial points.
 * A loose heuristic is to provide MOE with :math:`2D` historical points, where *D* is the dimension of the space MOE is searching over. MOE will still function with less points, but it will be primarily exploring (vs exploiting) as it bootstraps itself and learns information about the space.

.. _stencil: http://en.wikipedia.org/wiki/Stencil_(numerical_analysis)

How many function evaluations do I need before MOE is "done"?
-------------------------------------------------------------

This is highly dependent on the dimension of the space that is being searched over, the size of the domain relative to the length scale in each dimension, and how "well behaved" the underlying objective function is.

One can:

 * Run MOE until the difference between consecutive suggested points falls below some threshold.
 * Run MOE for a fixed number of iterations. MOE will optimize the Expected Improvement at every evaluation, so whenever you stop you can know that you have sampled the points of highest Expected Improvement given your sample constraints.
 * A (very) loose heuristic is to sample `10D` historical points, where *D* is the dimension of the space MOE is searching over.

How many function evaluations do I perform before I update the hyperparameters of the GP?
-----------------------------------------------------------------------------------------

This is also highly dependent on the problem, but a good loose heuristic is every 5-10 historical points sampled.

Will you accept my pull request?
--------------------------------

Yes! Please follow the guidelines at :doc:`contributing`. Bonus points if you are addressing an `open issue`_.

.. _open issue: https://github.com/Yelp/MOE/issues
