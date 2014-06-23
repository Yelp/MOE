Welcome to MOE's documentation!
===============================

**Contents:**

    #. `Github repo`_
    #. `What is MOE?`_
    #. `Quick Install`_ and :doc:`Full Install </install>`
    #. `Quick Start`_
    #. `Source Documentation`_
    #. :doc:`Contributing </contributing>`

.. _Github repo: https://github.com/sc932/MOE

What is MOE?
-----------

MOE (Metric Optimization Engine) is a *fast and efficient*, *derivative-free*,  *black box*, *global* optimization framework for optimizing parameters of time *consuming* or *expensive* experiments and systems.

An experiment or system can be time consuming or expensive if it takes a long time to recieve statistically significant results (traffic for an A/B test, complex system with long training time, etc) or the opportunity cost of trying new values is high (engineering expense, A/B testing tradeoffs, etc).

MOE solves this problem through optimal experimental design and *optimal learning*.

    "Optimal learning addresses the challenge of how to collect information as efficiently as possible, primarily for settings where collecting information is time consuming and expensive"

    -- Prof. Warren Powell, http://optimallearning.princeton.edu

It boils down to:

    "What is the most efficient way to collect information?"

    -- Prof. Peter Frazier, http://people.orie.cornell.edu/pfrazier

The *black box* nature of MOE allows us to optimize any number of systems, requiring no internal knowledge or access. It uses some :doc:`objective function </objective_functions>` and some set of :doc:`parameters </objective_functions>` and finds the best set of parameters to maximize (or minimize) the given function in as few attempts as possible. It does not require knowledge of the specific objective, or how it is obtained, just the previous parameters and their associated objective values (historical data).

Example:

.. math::

    \underset{\vec{x}}{\mathrm{argmax}} \ \text{CTR} (\vec{x})

Where :math:`\vec{x}` is any real valued input in some finite number of dimensions and CTR is some black box function that is difficult, expensive or time consuming to evaluate and is potentially non-convex, non-differentiable or non-continuous.

We want to find the best set of parameters :math:`\vec{x}` while evaluating the underlying function (CTR) as few times as possible. See :doc:`Objective Functions </objective_functions>` for more examples of objective functions and the best ways to combine metrics.

It allows you to build the following loop, contantly optimizing and trading off the exploration and exploitation of the underlying parameter space. By continuing to optimize over many iterations MOE readily finds maxima in the objective function optimally (climbing the mountains of traditional optimization). By sampling and optimizing over many iterations of the MOE loop in time, we can also allow to surf these shifting optima as features and the world change in time. MOE surfs these waves of optima, attempting to stay at the peak of the potentially changing objective function in parameter space as time advances.

.. image:: ../moe/static/img/moe_loop.png
    :align: center
    :alt: moe loop
    :scale: 100%

For more examples on how MOE can be used see :doc:`examples`

Video and slidedeck introduction to MOE:

    * `15 min MOE intro video`_
    * `MOE intro slides`_

.. _15 min MOE intro video: http://www.youtube.com/watch?v=qAN6iyYPbEE

.. _MOE intro slides: http://www.slideshare.net/YelpEngineering/yelp-engineering-open-house-112013-optimally-learning-for-fun-and-profit


MOE does this internally by:

1. Building a Gaussian Process (GP) with the historical data

    - :doc:`gpp_math`
    - :mod:`moe.views.rest.gp_mean_var`
    - `RW Chapter 2`_

2. Optimizing the hyperparameters of the Gaussian Process (model selection)

    - :doc:`gpp_covariance`
    - :doc:`gpp_model_selection_and_hyperparameter_optimization`
    - :mod:`moe.views.rest.gp_hyper_opt`
    - `RW Chapter 4`_
    - `RW Chapter 5`_

3. Finding the points of highest Expected Improvement (EI)

    - :doc:`gpp_expected_improvement_demo`
    - :mod:`moe.views.rest.gp_ei`
    - `EGO Paper`_

4. Returning the points to sample, then repeat

.. _RW Chapter 2: http://www.gaussianprocess.org/gpml/chapters/RW2.pdf

.. _RW Chapter 4: http://www.gaussianprocess.org/gpml/chapters/RW4.pdf

.. _RW Chapter 5: http://www.gaussianprocess.org/gpml/chapters/RW5.pdf

.. _EGO Paper: http://www.ressources-actuarielles.net/EXT/ISFA/1226.nsf/0/f84f7ac703bf5862c12576d8002f5259/$FILE/Jones98.pdf

Externally you can use MOE through:

    * :doc:`The REST interface </moe.views.rest>`
    * :doc:`The python interface </moe.optimal_learning.python.python_version>`
    * :doc:`The C++ interface </cpp_tree>`
    * The CUDA kernels.

You can be up and optimizing in a matter of minutes.


Quick Install
----

Install in docker:
....

This is the recommended way to run the MOE REST server. All dependencies and building is done automatically and in an isolated container.

Docker (http://docs.docker.io/) is a container based virtualization framework. Unlike traditional virtualization Docker is fast, lightweight and easy to use. Docker allows you to create containers holding all the dependencies for an application. Each container is kept isolated from any other, and nothing gets shared.

::

    $ git clone https://github.com/sc932/MOE.git
    $ cd MOE
    $ docker build -t moe_container .
    $ docker run -p 6543:6543 moe_container

The webserver and REST interface is now running on port 6543 from within the container.

Build from source (linux and OSX 10.8 and 10.9 supported)
....

:doc:`Full Install </install>`

Quick Start
-----------

REST/web server and interactive demo
........

To get the REST server running locally, from the directory MOE is installed:

::

    $ pserve --reload development.ini

In your favorite browser go to: http://127.0.0.1:6543/

Or, from the command line,

::

    $ curl -X POST -H "Content-Type: application/json" -d '{"domain_info": {"dim": 1}, "points_to_evaluate": [[0.1], [0.5], [0.9]], "gp_info": {"points_sampled": [{"value_var": 0.01, "value": 0.1, "point": [0.0]}, {"value_var": 0.01, "value": 0.2, "point": [1.0]}]}}' http://127.0.0.1:6543/gp/ei

``gp_ei`` endpoint documentation: :mod:`moe.views.rest.gp_ei`

From ipython
....

::

    $ ipython
    > from moe.easy_interface.experiment import Experiment
    > from moe.easy_interface.simple_endpoint import gp_next_points
    > exp = Experiment([[0, 2], [0, 4]])
    > exp.add_point([0, 0], 1.0, 0.01)
    > next_point_to_sample = gp_next_points(exp)
    > print next_point_to_sample

``easy_interface`` documentation: :doc:`moe.easy_interface`

Within python
....

.. code-block:: python

    from moe.easy_interface.experiment import Experiment
    from moe.easy_interface.simple_endpoint import gp_next_points

    import math, random
    def function_to_minimize(x):
        """This function has a minimum near [1, 2.6]."""
        return math.sin(x[0]) * math.cos(x[1]) + math.cos(x[0] + x[1]) + random.uniform(-0.02, 0.02)

    exp = Experiment([[0, 2], [0, 4]])
    exp.add_point([0, 0], 1.0, 0.01) # Bootstrap with some known or already sampled point

    # Sample 20 points
    for i in range(20):
        next_point_to_sample = gp_next_points(exp)[0] # By default we only ask for one point
        value_of_next_point = function_to_minimize(next_point_to_sample)
        exp.add_point(next_point_to_sample, value_of_next_point, 0.01) # We can add some noise

    print exp.best_point

Source Documentation
====================

Documentation
-------------

.. toctree::
   :maxdepth: 2

   why_moe.rst
   install.rst
   objective_functions.rst
   examples.rst
   contributing.rst

Python Files
------------

.. toctree::
   :maxdepth: 4

   moe

C++ Files
---------

.. toctree::
   :maxdepth: 3

   cpp_tree.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
