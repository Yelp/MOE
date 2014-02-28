MOE
===

Metric Optimization Engine.

Install
-------

Requires:

1. `python 2.6.7+` - http://python.org/download/
2. `gcc 4.7.3+` - http://gcc.gnu.org/install/
3. `cmake 2.8.9+` - http://www.cmake.org/cmake/help/install.html
4. `boost 1.51+` - http://www.boost.org/users/download/
5. `pip 1.2.1+` - http://pip.readthedocs.org/en/latest/installing.html
6. `mongodb 2.4.9+` - http://docs.mongodb.org/manual/installation/
7. We recommend using a virtualenv http://www.jontourage.com/2011/02/09/virtualenv-pip-basics/

Install:

```bash
$ git clone https://github.com/sc932/MOE.git
$ cd MOE
$ git pull origin master
$ make
$ pip install -e .
$ pserve --reload development.ini
```

OSX tips:

Download macports - http://www.macports.org/install.php

Download xQuartz (needed for X11, needed for matplotlob) - http://xquartz.macosforge.org/landing/

Getting gcc and boost and matplotlib reqs:
```bash
$ sudo port selfupdate
$ sudo port install gcc47
$ sudo port select --set gcc mp-gcc47
$ sudo port install boost
$ sudo port install py-matplotlib
```

Contributing
------------

1. Fork it.
2. Create a branch (`git checkout -b my_moe_branch`) (make sure to run `make test`)
3. Commit your changes (`git commit -am "Added Some Mathemagics"`)
4. Push to the branch (`git push origin my_moe_branch`)
5. Open a [Pull Request][1]
6. Optimize locally while you wait

[1]: http://github.com/sc932/MOE/pulls
