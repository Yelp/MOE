.. MOE documentation master file, created by
   sphinx-quickstart on Tue Mar 11 16:34:26 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MOE's documentation!
===============================

**Contents:**

    1. `Quick Start`_
    2. `Source Documentation`_

Quick Start
-----------

Requires
........

1. ``python 2.6.7+`` - http://python.org/download/
2. ``gcc 4.7.3+`` - http://gcc.gnu.org/install/
3. ``cmake 2.8.9+`` - http://www.cmake.org/cmake/help/install.html
4. ``boost 1.51+`` - http://www.boost.org/users/download/
5. ``pip 1.2.1+`` - http://pip.readthedocs.org/en/latest/installing.html
6. ``doxygen 1.8.5+`` - http://www.stack.nl/~dimitri/doxygen/index.html
7. We recommend using a virtualenv http://www.jontourage.com/2011/02/09/virtualenv-pip-basics/

Install
.......

Read "CMake Tips" (below) first if you are unfamiliar with CMake. OS X
users in particular will probably need to modify their ``cmake`` command.

::

    $ git clone https://github.com/sc932/MOE.git
    $ cd MOE
    $ git pull origin master
    $ mkdir build
    $ cd build
    $ cmake ../optimal_learning/EPI/src/cpp `# This line may require extra options and/or env vars.`
    $ cd ..
    $ make
    $ pip install -e .
    $ pserve --reload development.ini

CMake Tips
........
1. Do you have dependencies installed in non-standard places? e.g., did you build your own boost? Pass the ``-DCMAKE_FIND_ROOT_PATH`` option to cmake: ``cmake -DCMAKE_FIND_ROOT_PATH=/path/to/stuff ...`` (OS X users with MacPorts should set ``/opt/local``)
2. Are you using the right compiler? e.g., for ``gcc``, prepend ``CC=gcc CXX=g++``: ``CC=gcc CXX=g++ cmake ...`` (OS X users need to explicitly set this.)
3. Once you run cmake, the compiler is set in stone. If you mis-specify
   the compiler, run ``rm -fr *`` in ``build/`` before re-running
   ``cmake`` (the build-tree must be empty).

   See: http://www.cmake.org/Wiki/CMake_FAQ#How_do_I_use_a_different_compiler.3F
4. Using OS X with MacPorts? Your ``cmake`` command should probably read: ``CC=gcc CXX=g++ cmake -DCMAKE_FIND_ROOT_PATH=/opt/local ../optimal_learning/EPI/src/cpp``

OSX Tips
........

0. Are you sure you wouldn't rather be running linux?
1. Download MacPorts - http://www.macports.org/install.php (If you change the install directory from ``/opt/local``, don't forget to update the cmake invocation.)
2. MacPorts can resolve most dependencies. Make sure you set your ``PATH`` env var.
3. Download xQuartz (needed for X11, needed for matplotlib) - http://xquartz.macosforge.org/landing/ (Also available through MacPorts, see item 4.)
4. Getting gcc, boost, matplotlib, and xQuartz (``xorg-server``) reqs (before installing MOE):

::

    $ sudo port selfupdate
    $ sudo port install gcc47
    $ sudo port select --set gcc mp-gcc47
    $ sudo port install boost
    $ sudo port install xorg-server
    $ sudo port install py-matplotlib
    $ sudo port install doxygen

More OSX Tips
*************

1. Make sure you create your virtualenv with the correct python ``--python=/opt/local/bin/python`` if you are using MacPorts.

Linux Tips
..........

1. You can apt-get everything you need. Woo real package managers!
2. Having trouble with matplotlib dependencies? ``sudo apt-get install python-matplotlib``

Contributing
------------

1. Fork it.
2. Create a branch (``git checkout -b my_moe_branch``) (make sure to run ``make test``)
3. Commit your changes (``git commit -am "Added Some Mathemagics"``)
4. Push to the branch (``git push origin my_moe_branch``)
5. Open a `Pull Request <http://github.com/sc932/MOE/pulls>`_
6. Optimize locally while you wait

Source Documentation
====================

Python Files
------------

.. toctree::
   :maxdepth: 3

   optimal_learning.EPI.src.python

C++ Files
---------

.. toctree::
   :maxdepth: 3

   gpp_math_hpp.rst
   gpp_geometry_hpp.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

