Install
======

Install in docker:
-----

This is the recommended way to run the MOE REST server. All dependencies and building is done automatically and in an isolated container.

Docker (http://docs.docker.io/) is a container based virtualization framework. Unlike traditional virtualization Docker is fast, lightweight and easy to use. Docker allows you to create containers holding all the dependencies for an application. Each container is kept isolated from any other, and nothing gets shared.

::

    $ git clone https://github.com/sc932/MOE.git
    $ cd MOE
    $ docker build -t moe_container .
    $ docker run -p 6543:6543 moe_container

The webserver and REST interface is now running on port 6543 from within the container.

Install from source:
-----

Requires:

1. ``python 2.6.7+`` - http://python.org/download/
2. ``gcc 4.7.3+`` - http://gcc.gnu.org/install/
3. ``cmake 2.8.9+`` - http://www.cmake.org/cmake/help/install.html
4. ``boost 1.51+`` - http://www.boost.org/users/download/
5. ``pip 1.2.1+`` - http://pip.readthedocs.org/en/latest/installing.html
6. ``doxygen 1.8.5+`` - http://www.stack.nl/~dimitri/doxygen/index.html
7. We recommend using a virtualenv http://www.jontourage.com/2011/02/09/virtualenv-pip-basics/

::

    $ git clone https://github.com/sc932/MOE.git
    $ cd MOE
    $ pip install -e .
    $ python setup.py install

OSX Tips (<=10.8. For 10.9, see separate instructions below):
-----

0. Are you sure you wouldn't rather be running linux?
1. Download MacPorts - http://www.macports.org/install.php (If you change the install directory from ``/opt/local``, don't forget to update the cmake invocation.)
2. MacPorts can resolve most dependencies. Make sure you set your ``PATH`` env var.
3. Download xQuartz (needed for X11, needed for matplotlib) - http://xquartz.macosforge.org/landing/ (Also available through MacPorts, see item 4.)
4. Getting gcc, boost, matplotlib, and xQuartz (``xorg-server``) reqs (before installing MOE):
5. Make sure you create your virtualenv with the correct python ``--python=/opt/local/bin/python`` if you are using MacPorts
6. If you are using another package manager (like homebrew) you may need to modify ``opt/local`` below to point to your ``Cellar`` directory.
7. For the following commands, order matters, especially when selecting the proper gcc compiler.

::

    $ sudo port selfupdate
    $ sudo port install gcc47
    $ sudo port select --set gcc mp-gcc47
    $ sudo port install boost
    $ sudo port install xorg-server
    $ sudo port install py-matplotlib
    $ sudo port install doxygen
    $ export MOE_CMAKE_OPTS=-DCMAKE_FIND_ROOT_PATH=/opt/local && export MOE_CC_PATH=/opt/local/bin/gcc && export MOE_CXX_PATH=/opt/local/bin/g++

Additional Tips for 10.9:
^^^^

To ensure consistency, be sure to use full paths throughout the installation.

1. Currently, Boost should not be installed with MacPorts. You should build it from source (see section "Building Boost").
2. Boost, MOE, and the virtualenv must be built with the same python. We recommend using MacPorts Python: ``/opt/local/bin/python``. 

Under OS X 10.9, Apple switched their canonical C++ library from ``libstdc++`` (GNU) to ``libc++`` (LLVM); they are not ABI-compatible. To remain consistent, package managers are linking against ``libc++``. Since MOE is built with gcc, we need ``libstdc++``; thus dependencies must also be built with that C++ library. Currently, package managers do not have enough flexibility to operate several C++ libraries at once, and we do not expect this to change. Ignoring this condition leads to binary incompatibilities; e.g., see:
http://stackoverflow.com/questions/20134223/building-a-boost-python-application-on-macos-10-9-mavericks/

Building Boost:
^^^^

1. Download the Boost source (http://sourceforge.net/projects/boost/files/boost/1.55.0/ has been verfied to work).
2. From within the main directory, run (after checking additional options below):

::

    $ sudo ./bootstrap.sh --with-python=PYTHON
    $ sudo ./b2 install


2. Make sure ``which gcc`` is ``/opt/local/bin/gcc`` (macport installed) or whatever C++11 compliant gcc you want (similarly, ``which g++`` should be ``/opt/local/bin/g++``), and make sure Python is ``/opt/local/bin/python`` if using MacPorts or whichever Python you want to use. 
3. When building MOE, add to ``MOE_CMAKE_OPTS`` the ``BOOST_ROOT`` variable containing the location of the Boost that you have installed when running CMake and verify that CMake finds it (e.g., check a link.txt file in a ``moe/build/CMakeFiles/*.dir/`` dir and verify the location of ``libboost_python-mt`` or ``libboost_python``, whichever is appropriate)  
4. You might need to prepend ``BOOST_ROOT`` to ``CMAKE_FIND_ROOT_PATH=/opt/local`` to make this work if you have separate Boost installation(s). ``BOOST_ROOT`` is the ``path/to/your/boost_1_55_0``.

::

    $ export MOE_CMAKE_OPTS='-D BOOST_ROOT=/path/to/boost -D Boost_NO_SYSTEM_PATHS=ON -D CMAKE_FIND_ROOT_PATH=/path/to/boost:/opt/local'

5. If you elected to use a different Python than the one from MacPorts, make sure CMake is finding it (e.g., set the ``-DPYTHON_LIBRARIES=path/to/python.dylib`` env variable when running CMake). Check ``link.txt`` (see item above) to see if Python was found correctly.

Additional options for ``./boostrap.sh``:

1. ``--with-libraries=python,math,random,program_options,exception,system`` compiles only the libraries we need.
2. ``--prefix=path/to/install/dir`` builds Boost and pulls the libraries in the specified path. Default is ``/usr/local`` (recommended, especially if you already have system Boost installations; remember to set ``BOOST_ROOT``).

Additional options for ``./b2``: 

1. ``--build-dir=/path/to/build/dir`` builds the Boost files in a separate location instead of mixed into the source tree (recommended).
2. ``-j4`` uses 4 threads to compile (faster).

Linux Tips:
-----

1. You can apt-get everything you need. Yay for real package managers!

::

    $ apt-get update
    $ apt-get install python python-dev gcc cmake libboost-all-dev python-pip doxygen libblas-dev liblapack-dev gfortran git

CMake Tips:
-----

1. Do you have dependencies installed in non-standard places? e.g., did you build your own boost? Set the env var: ``export MOE_CMAKE_OPTS=-DCMAKE_FIND_ROOT_PATH=/path/to/stuff ...`` (OS X users with MacPorts should set ``/opt/local``) This can be used to set any number of cmake arguments.
2. Are you using the right compiler? e.g., for ``gcc``, run ``export MOE_CC_PATH=gcc && export MOE_CXX_PATH=g++`` (OS X users need to explicitly set this.)

