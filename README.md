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
7. we recommend using a virtualenv http://www.jontourage.com/2011/02/09/virtualenv-pip-basics/

Install:

```bash
$ git clone https://github.com/sc932/MOE.git
$ cd MOE
$ git pull origin master
$ make
$ pip install -e .
$ pserve --reload development.ini
