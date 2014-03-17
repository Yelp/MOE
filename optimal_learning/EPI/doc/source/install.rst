Installing EPI
==============

1. Download the source using git_ from github::

    git clone https://github.com/sc932/EPI.git

2. Move into the src directory and compile::

    cd EPI/src
    make

3. EPI can now be run::

    ./EPI [-options]

   or::

       ./EPI_runner.py [-options]

4. (optional) Build this documentation::

        cd ../doc
        make html

  the documentation can be found in build/html

.. _git: http://git-scm.com/

Dependencies
------------

1. python_ 2.7+

2. gcc_ (or another C compiler)

.. _python: http://www.python.org/
.. _gcc: http://gcc.gnu.org/

Python Package Dependencies
###########################

We suggest using a package manager such as pip_ (linux/osx) or brew_ (osx)

1. numpy_::

    sudo pip install numpy

2. scipy_

    on linux::

        sudo pip install scipy

    on OSX::

        http://www.scipy.org/Installing_SciPy/Mac_OS_X

3. matplotlib_

    on linux::

        sudo pip install matplotlib

    on OSX::

        sudo brew install pkgconfig
        cd Desktop/
        git clone git://github.com/matplotlib/matplotlib.git
        cd matplotlib/
        python setup.py build
        sudo python setup.py install

4. sphinx_ (optional for documentation)::

    sudo pip install sphinx

.. _pip: http://www.pip-installer.org/en/latest/
.. _brew: http://mxcl.github.com/homebrew/
.. _numpy: http://numpy.scipy.org/
.. _scipy: http://www.scipy.org/
.. _matplotlib: http://matplotlib.sourceforge.net/
.. _sphinx: http://sphinx.pocoo.org/


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
