MOE
===

Metric Optimization Engine.

Install
-------

Requires:
1. pip - http://pip.readthedocs.org/en/latest/installing.html
2. mongodb - http://docs.mongodb.org/manual/installation/
3. we recommend using a virtualenv http://www.jontourage.com/2011/02/09/virtualenv-pip-basics/

Install:
...
$ git clone https://github.com/sc932/MOE.git
$ cd MOE
$ git pull origin master
$ make
$ pip install -e .
$ pserve --reload development.ini
...
