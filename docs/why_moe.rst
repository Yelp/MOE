Why Do We Need MOE?
===================

MOE is designed for optimizing a system's parameters, when evaluating parameters is *time-consuming* or *expensive*, the objective function is a *black box* and not necessarily concave or convex, derivatives are unavailable, and we wish to find a global optimum, rather than a local one.

Finding an optimum in these situations manually is very difficult, because the number of possible sets of parameters to try can be so large, and the time required for each parameter evaluation makes it difficult to maintain focus over the required period of time.  Difficulties are compounded when the number of parameters being optimized over grows larger than two or three, making it difficult to visualize how the objective function varies with the parameters.

Using a simple hill-climbing method is also difficult because the lack of convexity means that there can be many local hills and valleys, causing such methods to get stuck in a local optimum.  Moreover, the lack of availability of derivatives means that just finding a direction of improvement to implement a hill-climbing approach can require many function evaluations.

One can use a heuristic optimization method like a genetic algorithm or simulated annealing, but these require parameters of their own to be set for them to work well, and many heuristics need to evaluate the objective function thousands or tens of thousands of times before finding an approximate optimum, which is infeasible when objective function evaluations are expensive.

All of these problems just get worse the more expensive or time-consuming our objective function becomes.

MOE solves all of these problems in an optimal way. See :doc:`examples` and :doc:`objective_functions`.
