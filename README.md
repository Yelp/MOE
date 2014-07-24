[![MOE logo](https://github.com/yelp/MOE/moe/static/img/moe_logo_48.png)
[![Build Status](https://magnum.travis-ci.com/Yelp/MOE.svg?token=E3yRnCAkWnWzepuxbk6A&branch=master)](https://magnum.travis-ci.com/Yelp/MOE)

Metric Optimization Engine. A global, black box optimization engine for real world metric optimization.

  * [Full documentation][1]
  * [REST documentation][2]

Or, build the documentation locally with `make docs`.

## What is MOE?

MOE (Metric Optimization Engine) is an *efficient* way to optimize a system's parameters, when evaluating parameters is *time-consuming* or *expensive*.

Here are some examples of when you could use MOE:

* **Optimizing a system's click-through rate (CTR).**  MOE is useful when evaluating CTR requires running an A/B test on real user traffic, and getting statistically significant results requires running this test for a substantial amount of time (hours, days, or even weeks).

* **Optimizing tunable parameters of a machine-learning prediction method.**  MOE is useful if calculating the prediction error for one choice of the parameters takes a long time, which might happen because the prediction method is complex and takes a long time to train, or because the data used to evaluate the error is huge.

* **Optimizing the design of an engineering system** (an airplane, the traffic network in a city, a combustion engine, a hospital).  MOE is useful if evaluating a design requires running a complex physics-based numerical simulation on a supercomputer. 

* **Optimizing the parameters of a real-world experiment** (a chemistry, biology, or physics experiment, a drug trial).  MOE is useful when every experiment needs to be physically created in a lab, or very few experiments can be run in parallel.

MOE is ideal for problems in which the optimization problem's objective function is a black box, not necessarily convex or concave, derivatives are unavailable, and we seek a global optimum, rather than just a local one. This ability to handle black-box objective functions allows us to use MOE to optimize nearly any system, without requiring any internal knowledge or access. To use MOE, we simply need to specify some [objective function][14], some set of [parameters][15], and any historical data we may have from previous evaluations of the objective function. MOE then finds the set of parameters that maximize (or minimize) the objective function, while evaluating the objective function as little as possible. 

Inside, MOE uses *Bayesian global optimization*, which performs optimization using Bayesian statistics and *optimal learning*. 

Optimal learning is the study of efficient methods for collecting information, particularly when doing so is time-consuming or expensive, and was developed and popularized from its roots in decision theory by [Prof. Peter Frazier][16] ([Cornell, Operations Research and Information Engineering][17]) and [Prof. Warren Powell][18] ([Princeton, Operations Research and Financial Engineering][19]). For more information about the mathematics of optimal learning, and more real-world applications like heart surgery, drug discovery, and materials science, see these [intro slides][20] to optimal learning.

[Why do we need MOE?][21]

Video and slidedeck introduction to MOE:

* [15 min MOE intro video][10]
* [MOE intro slides][11]
* [Full documentation][1]

MOE does this internally by:

1. Building a Gaussian Process (GP) with the historical data
2. Optimizing the hyperparameters of the Gaussian Process (model selection)
3. Finding the points of highest Expected Improvement (EI)
4. Returning the points to sample, then repeat

Externally you can use MOE through:

1. [The REST interface][2]
2. [The python interface][9]
3. [The C++ interface][12]

You can be up and optimizing in a matter of minutes. [Examples of using MOE][13]

## Running MOE

### REST/web server and interactive demo

from the directory MOE is installed:

```bash
$ pserve --reload development.ini # MOE server is now running at http://localhost:6543
```

[The REST interface documentation][2]

Or, from the command line,

```bash
$ curl -X POST -H "Content-Type: application/json" -d '{"domain_info": {"dim": 1}, "points_to_evaluate": [[0.1], [0.5], [0.9]], "gp_info": {"points_sampled": [{"value_var": 0.01, "value": 0.1, "point": [0.0]}, {"value_var": 0.01, "value": 0.2, "point": [1.0]}]}}' http://127.0.0.1:6543/gp/ei
```
[`gp_ei` endpoint documentation.][4]

### From ipython

```bash
$ ipython
> from moe.easy_interface.experiment import Experiment
> from moe.easy_interface.simple_endpoint import gp_next_points
> exp = Experiment([[0, 2], [0, 4]])
> exp.historical_data.append_sample_points([[0, 0], 1.0, 0.01])
> next_point_to_sample = gp_next_points(exp)
> print next_point_to_sample
```
[`easy_interface` documentation.][5]

### Within python

See ``examples/next_point_via_simple_endpoint.py`` for this code or http://yelp.github.io/MOE/examples.html for more examples.

```python
import math
import random

from moe.easy_interface.experiment import Experiment
from moe.easy_interface.simple_endpoint import gp_next_points
from moe.optimal_learning.python.data_containers import SamplePoint


# Note: this function can be anything, the output of a batch, results of an A/B experiment, the value of a physical experiment etc.
def function_to_minimize(x):
    """Calculate an aribitrary 2-d function with some noise with minimum near [1, 2.6]."""
    return math.sin(x[0]) * math.cos(x[1]) + math.cos(x[0] + x[1]) + random.uniform(-0.02, 0.02)

if __name__ == '__main__':
    exp = Experiment([[0, 2], [0, 4]])  # 2D experiment, we build a tensor product domain
    # Bootstrap with some known or already sampled point(s)
    exp.historical_data.append_sample_points([
        SamplePoint([0, 0], function_to_minimize([0, 0]), 0.05),  # Iterables of the form [point, f_val, f_var] are also allowed
        ])

    # Sample 20 points
    for i in range(20):
        # Use MOE to determine what is the point with highest Expected Improvement to use next
        next_point_to_sample = gp_next_points(exp)[0]  # By default we only ask for one point
        # Sample the point from our objective function, we can replace this with any function
        value_of_next_point = function_to_minimize(next_point_to_sample)

        print "Sampled f({0:s}) = {1:.18E}".format(str(next_point_to_sample), value_of_next_point)

        # Add the information about the point to the experiment historical data to inform the GP
        exp.historical_data.append_sample_points([SamplePoint(next_point_to_sample, value_of_next_point, 0.01)])  # We can add some noise
```

More examples can be found in the `<MOE_DIR>/examples` directory.

### Within C++

Expected Improvement Demo - http://yelp.github.io/MOE/gpp_expected_improvement_demo.html
Gaussian Process Hyperparameter Optimization Demo - http://yelp.github.io/MOE/gpp_hyperparameter_optimization_demo.html
Combined Demo - http://yelp.github.io/MOE/gpp_hyper_and_EI_demo.html

# Install

## Install in docker:

This is the recommended way to run the MOE REST server. All dependencies and building is done automatically and in an isolated container.

[Docker (http://docs.docker.io/)][6] is a container based virtualization framework. Unlike traditional virtualization Docker is fast, lightweight and easy to use. Docker allows you to create containers holding all the dependencies for an application. Each container is kept isolated from any other, and nothing gets shared.

```bash
$ git clone https://github.com/Yelp/MOE.git
$ cd MOE
$ docker build -t moe_container .
$ docker run -p 6543:6543 moe_container
```

The webserver and REST interface is now running on port 6543 from within the container.

## Install from source:

See [Install Documentation][7]

## Contributing

See [Contributing Documentation][8]

## License

MOE is licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0

[0]: https://www.youtube.com/watch?v=qAN6iyYPbEE
[1]: http://yelp.github.io/MOE/
[2]: http://yelp.github.io/MOE/moe.views.rest.html
[3]: http://github.com/Yelp/MOE/pulls
[4]: http://yelp.github.io/MOE/moe.views.rest.html#module-moe.views.rest.gp_ei
[5]: http://yelp.github.io/MOE/moe.easy_interface.html
[6]: http://docs.docker.io/
[7]: http://yelp.github.io/MOE/install.html
[8]: http://yelp.github.io/MOE/contributing.html
[9]: http://yelp.github.io/MOE/moe.optimal_learning.python.python_version.html
[10]: http://www.youtube.com/watch?v=qAN6iyYPbEE
[11]: http://www.slideshare.net/YelpEngineering/yelp-engineering-open-house-112013-optimally-learning-for-fun-and-profit
[12]: http://yelp.github.io/MOE/cpp_tree.html
[13]: http://yelp.github.io/MOE/examples.html
[14]: http://yelp.github.io/MOE/objective_functions.html
[15]: http://yelp.github.io/MOE/objective_functions.html#parameters
[16]: http://people.orie.cornell.edu/pfrazier/
[17]: http://www.orie.cornell.edu/
[18]: http://optimallearning.princeton.edu/
[19]: http://orfe.princeton.edu/
[20]: http://people.orie.cornell.edu/pfrazier/Presentations/2014.01.Lancaster.BGO.pdf
[21]: http://yelp.github.io/MOE/why_moe.html
