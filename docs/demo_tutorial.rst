Demo Tutorial
=============

MOE provides a Graphical User Interface for exploration of the various endpoints and components of the underlying system. After going through the :doc:`install` in docker this interface is running at http://localhost:6543.

.. image:: ../moe/static/img/moe_gui_index.png
    :align: center
    :alt: moe gui index
    :scale: 100%

This includes link to:

    #. All `docs`_ and `the repo`_
    #. :doc:`pretty_endpoints`
    #. The demo, which will be covered here.

.. _docs: http://sc932.github.io/MOE
.. _the repo: http://www.github.com/sc932/MOE

The Interactive Demo
--------------------

The GUI includes a demo for visualizing Gaussian Processes written in d3 and using various endpoints from the REST interface.

.. image:: ../moe/static/img/moe_demo_start.png
    :align: center
    :alt: moe demo start
    :scale: 100%

This view has several components (check out the tooltips for specific components):

 * The graph labeled **Gaussian Process (GP)** is the posterior mean and variance of the GP given the historical data and parameters (on right). The dashed line is the posterior mean, the faded area is the variance, for each point in [0,1].
 * The graph labeled **Expected Improvement (EI)** is a plot of EI for each potential next point to sample in [0,1]. The red line corresponds to the point of highest EI within the domain. This is the point that MOE suggest we sample next to optimize the EI.
 * On the right are various hyperparameters of the GP covariance and the parameters of the Stochastic Gradient Descent algorithm that we use to optimize EI.
 * On the bottom right we can specify new historical points to update the GP. By default the GUI suggests the point of highest EI and generates a value for the point drawn from the GP prior.

By sampling several points we can see the plots of GP and EI evolving.

.. image:: ../moe/static/img/moe_demo_one_point.png
    :align: center
    :alt: moe demo start
    :scale: 100%
