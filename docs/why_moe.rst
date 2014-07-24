Why Do We Need MOE?
===================

MOE is designed for optimizing a system's parameters, when evaluating parameters is *time-consuming* or *expensive*, the objective function is a *black box* and not necessarily concave or convex, derivatives are unavailable, and we wish to find a global optimum, rather than a local one.

Finding an optimum in these situations manually is very difficult, because the number of possible sets of parameters to try can be so large, and the time required for each parameter evaluation makes it difficult to maintain focus over the required period of time.  Difficulties are compounded when the number of parameters being optimized over grows larger than two or three, making it difficult to visualize how the objective function varies with the parameters.

Using a simple hill-climbing method is also difficult because the lack of convexity means that there can be many local hills and valleys, causing such methods to get stuck in a local optimum.  Moreover, the lack of availability of derivatives means that just finding a direction of improvement to implement a hill-climbing approach can require many function evaluations.

One can use a heuristic optimization method like a `genetic algorithm`_ or `simulated annealing`_, but these require parameters of their own to be set for them to work well, and many heuristics need to evaluate the objective function thousands or tens of thousands of times before finding an approximate optimum, which is infeasible when objective function evaluations are expensive.

All of these problems just get worse the more expensive or time-consuming our objective function becomes.

MOE solves all of these problems in an optimal way. See :doc:`examples` and :doc:`objective_functions`.

Other Methods
-------------

Many optimization methods exist for different types of problems:

* If derivative information is available and the system is noise-free and only a local optima is desired then a hill climbing algorithm like `Gradient Descent`_  can be used, or depending on the properties of the system one can use `Conjugate Gradient`_. If the system is noisy a method like `Stochastic Gradient Descent`_ can be used. MOE uses `Stochastic Gradient Descent`_ internally when finding the next point(s) of highest Expected Improvement (EI) of the Gaussian Process.
.. _Gradient Descent: http://en.wikipedia.org/wiki/Gradient_descent
.. _Conjugate Gradient: http://en.wikipedia.org/wiki/Conjugate_gradient_method
.. _Stochastic Gradient Descent: http://en.wikipedia.org/wiki/Stochastic_gradient_descent

* If function evaluation is inexpensive and not time-consuming and a local optima is desired without derivative information then a method that approximates derivates can be used like `BFGS`_, or a method that requires no derivatives like `Powell's method`_ or the `Nelder-Mead method`_.
.. _BFGS: http://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm
.. _Powell's method: http://en.wikipedia.org/wiki/Powell's_method
.. _Nelder-Mead method: http://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method

* If function evaluation is inexpensive and not time-consuming and a global optima is desired without derivative information then a method like `simulated annealing`_, `particle swarm optimization`_, or a `genetic algorithm`_ can be used. Note that these methods can take many more function evaluations than other listed methods to find an optima.
.. _simulated annealing: http://en.wikipedia.org/wiki/Simulated_annealing
.. _particle swarm optimization: http://en.wikipedia.org/wiki/Particle_swarm_optimization
.. _genetic algorithm: http://en.wikipedia.org/wiki/Genetic_algorithm

There are many optimization software packages that help solve these different problems:

* Many of the optimization methods listed above are implemented in numerical packages like `scipy.optimize`_.
.. _scipy.optimize: http://docs.scipy.org/doc/scipy/reference/optimize.html

* `DiceOptim`_ implements many similar methods to MOE, including Expected Improvement optimization using Gaussian Processes in R.
.. _DiceOptim: http://cran.r-project.org/web/packages/DiceOptim/index.html

* `space`_ implements the Efficient Global Optimization (EGO) method in C++.
.. _space: http://www.schonlau.net/space.html

* `spearmint`_ and `bayesopt`_ are both implementations of Bayesian Global Optimization and similar methods.
.. _spearmint: https://github.com/JasperSnoek/spearmint
.. _bayesopt: http://rmcantin.bitbucket.org/html/index.html

* `OpenTuner`_ is a package that combines many different methods and finds global optima when evaluation is inexpensive and not time-consuming.
.. _OpenTuner: http://opentuner.org/

* `hyperopt`_ is a package for optimizing hyperparameters that uses grid search and the Tree of Parzen Estimators method.
.. _hyperopt: https://github.com/hyperopt/hyperopt
