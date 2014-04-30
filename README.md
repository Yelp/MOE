# MOE

Metric Optimization Engine.

[Full documentation here.][2]

[2]: http://sc932.github.io/MOE/

Or, build the documentation locally with `make docs`.

## Running MOE

### REST/web server and interactive demo

from the directory MOE is installed:

```bash
$ pserve --reload development.ini
```

In your favorite browser go to: http://127.0.0.1:6543/

OR

```bash
$ curl -X POST -H "Content-Type: application/json" -d '{"points_to_evaluate": [[0.06727463396075942], [0.5067300380945079], [0.9698763624056982], [0.6741416078606629], [0.3413945823872875], [0.8293462326458892], [0.1895850103202945], [0.29784241725123095], [0.7611434260204735], [0.4050181259320824]], "points_being_sampled": [], "gp_info": {"points_sampled": [{"value_var": 0.01, "value": -2.014556917682888, "point": [0.8356251271367201]}, {"value_var": 0.01, "value": -1.3556680509922945, "point": [0.5775274088974685]}, {"value_var": 0.01, "value": -0.17644452034270924, "point": [0.1299624124365485]}, {"value_var": 0.01, "value": 0.3125023458503953, "point": [0.02303611187965965]}, {"value_var": 0.01, "value": -0.5899125641251172, "point": [0.3938472181674687]}, {"value_var": 0.01, "value": -1.8568254250899945, "point": [0.9894680586912427]}, {"value_var": 0.01, "value": -1.0638344140121117, "point": [0.45444660991161895]}, {"value_var": 0.01, "value": -0.28576907668798884, "point": [0.20420919931329756]}, {"value_var": 0.01, "value": -1.568109287685418, "point": [0.6404744671911634]}, {"value_var": 0.01, "value": -1.8418398343184625, "point": [0.7168047658371041]}], "domain": [[0, 1]]}}' http://127.0.0.1:6543/gp/ei
```

### From ipython

```bash
$ ipython
> from moe.easy_interface.experiment import Experiment
> from moe.easy_interface.simple_endpoint import gp_next_points
> exp = Experiment([[0, 2], [0, 4]])
> exp.add_point([0, 0], 1.0, 0.01)
> next_point_to_sample = gp_next_points(exp)
> print next_point_to_sample
```

### Within python

```python
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
```

# Install

## Install in docker:

This is the recommended way to run the MOE REST server. All dependencies and building is done automatically and in an isolated container.

Docker (http://docs.docker.io/) is a container based virtualization framework. Unlike traditional virtualization Docker is fast, lightweight and easy to use. Docker allows you to create containers holding all the dependencies for an application. Each container is kept isolated from any other, and nothing gets shared.

```bash
$ git clone https://github.com/sc932/MOE.git
$ cd MOE
$ docker build -t moe_container
$ docker run -p 6543:6543 moe_container
```

The webserver and REST interface is now running on port 6543 from within the container.

## Install from source:

Requires:

1. `python 2.6.7+` - http://python.org/download/
2. `gcc 4.7.3+` - http://gcc.gnu.org/install/
3. `cmake 2.8.9+` - http://www.cmake.org/cmake/help/install.html
4. `boost 1.51+` - http://www.boost.org/users/download/
5. `pip 1.2.1+` - http://pip.readthedocs.org/en/latest/installing.html
6. `doxygen 1.8.5+` - http://www.stack.nl/~dimitri/doxygen/index.html
7. We recommend using a virtualenv http://www.jontourage.com/2011/02/09/virtualenv-pip-basics/

```bash
$ git clone https://github.com/sc932/MOE.git
$ cd MOE
$ pip install -e .
$ python setup.py install
```

### OSX Tips:

0. Are you sure you wouldn't rather be running linux?
1. Download MacPorts - http://www.macports.org/install.php (If you change the install directory from `/opt/local`, don't forget to update the cmake invocation.)
2. MacPorts can resolve most dependencies. Make sure you set your `PATH` env var.
3. Download xQuartz (needed for X11, needed for matplotlib) - http://xquartz.macosforge.org/landing/ (Also available through MacPorts, see item 4.)
4. Getting gcc, boost, matplotlib, and xQuartz (`xorg-server`) reqs (before installing MOE):
5. Make sure you create your virtualenv with the correct python `--python=/opt/local/bin/python` if you are using MacPorts
6. If you are using another package manager (like homebrew) you may need to modify `opt/local` below to point to your `Cellar` directory.

```bash
$ sudo port selfupdate
$ sudo port install gcc47
$ sudo port select --set gcc mp-gcc47
$ sudo port install boost
$ sudo port install xorg-server
$ sudo port install py-matplotlib
$ sudo port install doxygen
$ export MOE_CMAKE_OPTS=-DCMAKE_FIND_ROOT_PATH=/opt/local && export MOE_CC_PATH=/opt/local/bin/gcc && export MOE_CXX_PATH=/opt/local/bin/g++
```

### Linux Tips:

1. You can apt-get everything you need. Yay for real package managers!

```bash
$ apt-get update
$ apt-get install python python-dev gcc cmake libboost-all-dev python-pip doxygen libblas-dev liblapack-dev gfortran git
```

### CMake Tips:

1. Do you have dependencies installed in non-standard places? e.g., did you build your own boost? Set the env var: `export MOE_CMAKE_OPTS=-DCMAKE_FIND_ROOT_PATH=/path/to/stuff ...` (OS X users with MacPorts should set `/opt/local`) This can be used to set any number of cmake arguments.
2. Are you using the right compiler? e.g., for `gcc`, run `export MOE_CC_PATH=gcc && export MOE_CXX_PATH=g++` (OS X users need to explicitly set this.)


# Contributing

1. Fork it.
2. Create a branch (`git checkout -b my_moe_branch`)
3. Develop your feature/fix (don't forget to add tests!)
4. Run tests (`tox`)
5. Test against styleguide (`tox -e pep8 && tox -e pep257`)
6. Commit your changes (`git commit -am "Added Some Mathemagics"`)
7. Push to the branch (`git push origin my_moe_branch`)
8. Open a [Pull Request][1]
9. Optimize locally while you wait

[1]: http://github.com/sc932/MOE/pulls
