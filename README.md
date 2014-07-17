[![Build Status](https://magnum.travis-ci.com/sc932/MOE.svg?token=E3yRnCAkWnWzepuxbk6A&branch=master)](https://magnum.travis-ci.com/sc932/MOE)

# MOE

Metric Optimization Engine. A global, black box optimization engine for real world metric optimization.

  * [Full documentation][1]
  * [REST documentation][2]

Or, build the documentation locally with `make docs`.

## What is MOE?

MOE (Metric Optimization Engine) is a *fast and efficient*, *derivative-free*,  *black box*, *global* optimization framework for optimizing parameters of time *consuming* or *expensive* experiments and systems.

An experiment or system can be time consuming or expensive if it takes a long time to recieve statistically significant results (traffic for an A/B test, complex system with long training time, etc) or the opportunity cost of trying new values is high (engineering expense, A/B testing tradeoffs, etc).

MOE solves this problem through optimal experimental design and *optimal learning*.

> "Optimal learning addresses the challenge of how to collect information as efficiently as possible, primarily for settings where collecting information is time consuming and expensive"
> -- Prof. Warren Powell, http://optimallearning.princeton.edu

It boils down to:

> "What is the most efficient way to collect information?"
> -- Prof. Peter Frazier, http://people.orie.cornell.edu/pfrazier

The *black box* nature of MOE allows us to optimize any number of systems, requiring no internal knowledge or access. It uses some [objective function][14] and some set of [parameters][15] and finds the best set of parameters to maximize (or minimize) the given function in as few attempts as possible. It does not require knowledge of the specific objective, or how it is obtained, just the previous parameters and their associated objective values (historical data).

[Why do we need MOE?][16]

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
$ pserve --reload development.ini
```

In your favorite browser go to: http://127.0.0.1:6543/

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

See ``examples/next_point_via_simple_endpoint.py`` for this code or http://sc932.github.io/MOE/examples.html for more examples.

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

Expected Improvement Demo - http://sc932.github.io/MOE/gpp_expected_improvement_demo.html
Gaussian Process Hyperparameter Optimization Demo - http://sc932.github.io/MOE/gpp_hyperparameter_optimization_demo.html
Combined Demo - http://sc932.github.io/MOE/gpp_hyper_and_EI_demo.html

# Install

## Install in docker:

This is the recommended way to run the MOE REST server. All dependencies and building is done automatically and in an isolated container.

[Docker (http://docs.docker.io/)][6] is a container based virtualization framework. Unlike traditional virtualization Docker is fast, lightweight and easy to use. Docker allows you to create containers holding all the dependencies for an application. Each container is kept isolated from any other, and nothing gets shared.

```bash
$ git clone https://github.com/sc932/MOE.git
$ cd MOE
$ docker build -t moe_container .
$ docker run -p 6543:6543 moe_container
```

The webserver and REST interface is now running on port 6543 from within the container.

## Install from source:

See [Intall Documentation][7]

## Contributing

See [Contributing Documentation][8]

[0]: https://www.youtube.com/watch?v=qAN6iyYPbEE
[1]: http://sc932.github.io/MOE/
[2]: http://sc932.github.io/MOE/moe.views.rest.html
[3]: http://github.com/sc932/MOE/pulls
[4]: http://sc932.github.io/MOE/moe.views.rest.html#module-moe.views.rest.gp_ei
[5]: http://sc932.github.io/MOE/moe.easy_interface.html
[6]: http://docs.docker.io/
[7]: http://sc932.github.io/MOE/install.html
[8]: http://sc932.github.io/MOE/contributing.html
[9]: http://sc932.github.io/MOE/moe.optimal_learning.python.python_version.html
[10]: http://www.youtube.com/watch?v=qAN6iyYPbEE
[11]: http://www.slideshare.net/YelpEngineering/yelp-engineering-open-house-112013-optimally-learning-for-fun-and-profit
[12]: http://sc932.github.io/MOE/cpp_tree.html
[13]: http://sc932.github.io/MOE/examples.html
[14]: http://sc932.github.io/MOE/objective_functions.html
[15]: http://sc932.github.io/MOE/objective_functions.html#parameters
[16]: http://sc932.github.io/MOE/why_moe.html
