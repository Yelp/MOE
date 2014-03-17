MOE
===

Metric Optimization Engine.

[Full documentation here.][2]

[2]: http://sc932.github.io/MOE/

Install
-------

Requires:

1. `python 2.6.7+` - http://python.org/download/
2. `gcc 4.7.3+` - http://gcc.gnu.org/install/
3. `cmake 2.8.9+` - http://www.cmake.org/cmake/help/install.html
4. `boost 1.51+` - http://www.boost.org/users/download/
5. `pip 1.2.1+` - http://pip.readthedocs.org/en/latest/installing.html
6. `doxygen 1.8.5+` - http://www.stack.nl/~dimitri/doxygen/index.html
7. We recommend using a virtualenv http://www.jontourage.com/2011/02/09/virtualenv-pip-basics/

Install:

```bash
$ git clone https://github.com/sc932/MOE.git
$ cd MOE
$ git pull origin master
$ ./configure
$ make
$ pip install -e .
$ pserve --reload development.ini
```

OSX tips:

0. Are you sure you wouldn't rather be running linux?
1. Download macports - http://www.macports.org/install.php
2. Most dependencies macports can resolve, make sure you set your `PATH` env var.
3. Download xQuartz (needed for X11, needed for matplotlob) - http://xquartz.macosforge.org/landing/
4. Getting gcc and boost and matplotlib reqs (before installing MOE):

```bash
$ sudo port selfupdate
$ sudo port install gcc47
$ sudo port select --set gcc mp-gcc47
$ sudo port install boost
$ sudo port install py-matplotlib
$ sudo port install doxygen
```

More OSX tips:

1. Make sure you create your virtualenv with the correct python `--python=/opt/local/bin/python` if you are using macports

Linux tips:

1. You can apt-get everything you need. Woo real package managers!
2. Having trouble with matplotlib dependencies? `sudo apt-get install python-matplotlib`

Contributing
------------

1. Fork it.
2. Create a branch (`git checkout -b my_moe_branch`) (make sure to run `make test`)
3. Commit your changes (`git commit -am "Added Some Mathemagics"`)
4. Push to the branch (`git push origin my_moe_branch`)
5. Open a [Pull Request][1]
6. Optimize locally while you wait

[1]: http://github.com/sc932/MOE/pulls
