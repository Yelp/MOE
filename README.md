MOE
===

Metric Optimization Engine.

Install
-------

Requires:
pip - http://pip.readthedocs.org/en/latest/installing.html
mongodb - http://docs.mongodb.org/manual/installation/

Install:
git clone https://github.com/sc932/MOE.git
cd MOE
git pull origin master
make
pip install -e . (we recommend using a virtualenv http://www.jontourage.com/2011/02/09/virtualenv-pip-basics/)
pserve --reload development.ini
