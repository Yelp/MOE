# MOE

Metric Optimization Engine.

[Full documentation here.][2]

[2]: http://sc932.github.io/MOE/

## Running MOE

### REST/web server and interactive demo

from the directory MOE is installed:

```bash
$ pserve --reload development.ini
```

In your favorite browser go to: http://127.0.0.1:6543/

### CLI

```bash
$ ipython
> import moe.optimal_learning
> # Do Stuff
```

# Install

Requires:

1. `python 2.6.7+` - http://python.org/download/
2. `gcc 4.7.3+` - http://gcc.gnu.org/install/
3. `cmake 2.8.9+` - http://www.cmake.org/cmake/help/install.html
4. `boost 1.51+` - http://www.boost.org/users/download/
5. `pip 1.2.1+` - http://pip.readthedocs.org/en/latest/installing.html
6. `doxygen 1.8.5+` - http://www.stack.nl/~dimitri/doxygen/index.html
7. We recommend using a virtualenv http://www.jontourage.com/2011/02/09/virtualenv-pip-basics/

## Install:

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
4. Getting gcc, boost, matplotlib, and xQuartz (`xorg-server) reqs (before installing MOE):
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

1. You can apt-get everything you need. Woo real package managers!
2. Having trouble with matplotlib dependencies? `sudo apt-get install python-matplotlib`

### CMake Tips:

1. Do you have dependencies installed in non-standard places? e.g., did you build your own boost? Set the env var: `export MOE_CMAKE_OPTS=-DCMAKE_FIND_ROOT_PATH=/path/to/stuff ...` (OS X users with MacPorts should set `/opt/local`) This can be used to set any number of cmake arguments.
2. Are you using the right compiler? e.g., for `gcc`, run `export MOE_CC_PATH=gcc && export MOE_CXX_PATH=g++` (OS X users need to explicitly set this.)


# Contributing

1. Fork it.
2. Create a branch (`git checkout -b my_moe_branch`) (make sure to run `make test` and `pyflakes`/`jslint`)
3. Commit your changes (`git commit -am "Added Some Mathemagics"`)
4. Push to the branch (`git push origin my_moe_branch`)
5. Open a [Pull Request][1]
6. Optimize locally while you wait

[1]: http://github.com/sc932/MOE/pulls
