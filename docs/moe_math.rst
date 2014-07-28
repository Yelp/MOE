How does MOE work?
==================

**Steps toward optimization:**

    #. Build a Gaussian Process (GP) with the historical data
    #. Optimize the hyperparameters of the Gaussian Process
    #. Find the point(s) of highest Expected Improvement (EI)
    #. Return the point(s) to sample, then repeat

This section has been given as `a 15 minute talk`_ with `slides`_ as well.

.. _a 15 minute talk: http://www.youtube.com/watch?v=qAN6iyYPbEE
.. _slides: http://www.slideshare.net/YelpEngineering/yelp-engineering-open-house-112013-optimally-learning-for-fun-and-profit

This section will use examples generated from the :doc:`demo_tutorial` included with MOE.

Build a Gaussian Process (GP) with the historical data
------------------------------------------------------

.. image:: ../moe/static/img/moe_demo_hyper_default.png
    :align: center
    :alt: moe demo three points sampled
    :scale: 100%

The GP evolves as it is given more historical data, making a better and better model of the underlying function.

See:

    - :ref:`sampling-points`
    - :doc:`gpp_math`
    - :mod:`moe.views.rest.gp_mean_var`
    - `RW Chapter 2`_

Optimize the hyperparameters of the Gaussian Process
----------------------------------------------------

.. image:: ../moe/static/img/moe_demo_hyper_big_sig_var.png
    :align: center
    :alt: moe demo four points sampled
    :scale: 100%

This plot has the same historical data as the plot above. By optimizing the hyperparameters of the GP we select the best model of our underlying function (with respect to a choice of likelihood functions).

See:

    - :ref:`changing-hypers`
    - :doc:`gpp_covariance`
    - :doc:`gpp_model_selection`
    - :mod:`moe.views.rest.gp_hyper_opt`
    - `RW Chapter 4`_
    - `RW Chapter 5`_

Find the point(s) of highest Expected Improvement (EI)
------------------------------------------------------

MOE suggests the points of highest Expected Improvement be sampled next. These are the points that are expected to beat the current best sampled point by the most.

See:

    - :doc:`gpp_expected_improvement_demo`
    - :mod:`moe.views.rest.gp_ei`
    - `EGO Paper`_

Return the point(s) to sample, then repeat
------------------------------------------

By continuing to optimize over many iterations, MOE quickly finds approximate optima, or points with large CTR.  As the world changes over time, MOE can surf these shifting optima as they move, staying at the peak of the potentially changing objective function in parameter space as time advances.

.. image:: ../moe/static/img/moe_loop.png
    :align: center
    :alt: moe loop
    :scale: 100%

For more examples on how MOE can be used see :doc:`examples`, or see :doc:`why_moe` for information about how MOE is useful for these kinds of problems and some alternate methods.

.. _RW Chapter 2: http://www.gaussianprocess.org/gpml/chapters/RW2.pdf
.. _RW Chapter 4: http://www.gaussianprocess.org/gpml/chapters/RW4.pdf
.. _RW Chapter 5: http://www.gaussianprocess.org/gpml/chapters/RW5.pdf
.. _EGO Paper: http://www.ressources-actuarielles.net/EXT/ISFA/1226.nsf/0/f84f7ac703bf5862c12576d8002f5259/$FILE/Jones98.pdf
