Why Do We Need MOE?
===================

**Contents:**

    #. `What is MOE?`_
    #. `Why is this hard?`_

What is MOE?
------------

MOE (Metric Optimization Engine) is a *fast and efficient*, *derivative-free*,  *black box*, *global* optimization framework for optimizing parameters of time *consuming* or *expensive* experiments and systems.

An experiment or system can be time consuming or expensive if it takes a long time to recieve statistically significant results (traffic for an A/B test, complex system with long training time, etc) or the opportunity cost of trying new values is high (engineering expense, A/B testing tradeoffs, etc).

MOE solves this problem through optimal experimental design and *optimal learning*.

    "Optimal learning addresses the challenge of how to collect information as efficiently as possible, primarily for settings where collecting information is time consuming and expensive"

    -- Prof. Warren Powell, http://optimallearning.princeton.edu

It boils down to:

    "What is the most efficient way to collect information?"

    -- Prof. Peter Frazier, http://people.orie.cornell.edu/pfrazier

The *black box* nature of MOE allows us to optimize any number of systems, requiring no internal knowledge or access. It uses some `objective function </objective_functions>` and some set of `parameters </objective_functions>` and finds the best set of parameters to maximize (or minimize) the given function in as few attempts as possible. It does not require knowledge of the specific objective, or how it is obtained, just the previous parameters and their associated objective values (historical data).

Why is this hard?
-----------------

#. Parameter optimization is hard

    a. Finding the perfect set of parameters takes a long time (exponential with number of dimensions)
    b. We can hope it is well behaved and try to move in the right direction (assume convex)
    c. Not possible as number of parameters increases (takes a lot of engineering and wall-clock time)

#. Intractable to find best set of parameters in all situations

    a. Thousands of combinations of device/user/location/time tuples to optimize over
    b. Finding the best parameters manually is impossible

#. Heuristics quickly break down in the real world

    a. Dependent parameters (changes to one change all others)
    b. Many parameters at once (high dimensional)
    c. Non-linear (complexity and chaos break assumptions)

All of these problems just get worse the more expensive or time consuming the function we are optimizing is to evaluate.

MOE solves all of these problems in an optimal way. See :doc:`examples` and :doc:`objective_functions`.
