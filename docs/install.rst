Install
=======

Install in docker
-----------------

This is the recommended way to run the MOE REST server. All dependencies and building is done automatically and in an isolated container.

`Docker`_ is a container based virtualization framework. Unlike traditional virtualization Docker is fast, lightweight and easy to use. Docker allows you to create containers holding all the dependencies for an application. Each container is kept isolated from any other, and nothing gets shared. To launch a pre-made docker container running MOE:

.. _Docker: http://docs.docker.io/

::

    $ docker pull yelpmoe/latest # You can also pull specific versions like yelpmoe/v0.1.0
    $ docker run -p 6543:6543 yelpmoe/latest

.. Note:: ``docker pull yelpmoe/foo`` downloads a docker image from `DockherHub`_. This is independent from your local MOE directory and will not see any local changes.

.. _DockherHub: https://hub.docker.com/

If you are on OSX, or want a build based on the current master branch you may need to build this manually.

::

    $ git clone https://github.com/Yelp/MOE.git
    $ cd MOE
    $ docker build -t moe_container .
    $ docker run -p 6543:6543 moe_container

.. Note:: If you want a "stock" version of MOE, you must run these commands in a **clean** MOE repo (e.g., ``git clone`` into ``MOE_clean``). Unlike the ``docker pull`` use case above, this docker container will see any local changes in the directory from which ``docker build`` is run.
   
The webserver and REST interface is now running on port 6543 from within the container. http://localhost:6543

If you want to build a specific version of the container locally then use::

    $ git clone https://github.com/Yelp/MOE.git
    $ cd MOE
    $ git tag -l # lists all versions
    $ git checkout tags/v0.1.0 # or whatever version you want
    $ docker build -t moe_container_v0.1.0 .
    $ docker run -p 6543:6543 moe_container_v0.1.0

.. Note:: As with the previous example, this ``docker build`` will see local changes (e.g., files not checked into ``git``). If you want a "stock" build, you must run these commands in a **clean** MOE repo.

Install from source
-------------------

To ensure consistency, be sure to use full paths throughout the installation.

Requires:

1. ``python 2.6.7+`` - http://python.org/download/
2. ``gcc 4.7.3+`` - http://gcc.gnu.org/install/
3. ``cmake 2.8.9+`` - http://www.cmake.org/cmake/help/install.html
4. ``boost 1.51+`` - http://www.boost.org/users/download/
5. ``pip 1.2.1+`` - http://pip.readthedocs.org/en/latest/installing.html
6. ``doxygen 1.8.5+`` - http://www.stack.nl/~dimitri/doxygen/index.html

   .. _virtualenv quickstart:

7. We recommend using a ``virtualenv``: http://www.jontourage.com/2011/02/09/virtualenv-pip-basics/

   Install it with ``pip install virtualenv``. ``virtualenv`` requires that Python and pip are already installed. See `Linux Tips`_ or `OSX Tips`_ for tips on how to install these. Build your virtualenv with:

   ::

      # Using straight virtualenv
      $ virtualenv --no-site-packages --python=path/to/python ENV_NAME
      # Or using virtualenvwrapper
      $ mkvirtualenv --no-site-packages --python=path/to/python ENV_NAME

   The option ``--python`` is only necessary if you want to specify a version of Python different from the active Python (i.e., the result of ``which python``). OS X users in particular want to do this (see `OSX Tips`_).

8. After all the core requirements are installed, pip and MOE will handle the rest. Run these commands to clone MOE, build its python dependencies, and build MOE. These commands should preferably be run from the virtualenv you built in the `virtualenv quickstart`_:

   ::

      $ git clone https://github.com/Yelp/MOE.git
      $ cd MOE
      $ pip install -r requirements.txt
      $ python setup.py install

   .. Note:: MOE's ``setup.py`` invokes cmake. ``setup.py`` installs MOE with the python installation used to run it; so be sure to invoke ``setup.py`` with the Python that you want to use to run MOE. If this fails, then consult `Python Tips`_. Users can pass command line arguments to cmake via the ``MOE_CMAKE_OPTS`` environment variable. Other sections (e.g., `Python Tips`_, `CMake Tips`_) detail additional environment variables that may be needed to customize cmake's behavior.

   .. Warning:: Boost, MOE, and the virtualenv must be built with the same python. (OS X users: we recommend using the MacPorts Python: ``/opt/local/bin/python``.)

OSX Tips
--------

OS X 10.9 users beware: do not install boost with MacPorts. You *must* install it from source; see warnings below.

0. Are you sure you wouldn't rather be running linux?
1. Download `MacPorts`_. (If you change the install directory from ``/opt/local``, don't forget to update the cmake invocation.)
2. Read `General MacPorts Tips`_ if you are not familiar with MacPorts. MacPorts is one of many OS X package managers; we will use it to install MOE's core requirements.
3. MacPorts requires that your ``PATH`` variable include ``/opt/local/bin:/opt/local/sbin``. It sets this in your shell's ``rcfile`` (e.g., ``.bashrc``), but that command will not run immediately after MacPorts installation. So start a new shell or run ``export PATH=/opt/local/bin:/opt/local/sbin:$PATH``.
4. Make sure you create your virtualenv with the correct python ``--python=/opt/local/bin/python`` if you are using MacPorts.
5. If you are using another package manager (like homebrew) you may need to modify ``opt/local`` below to point to your ``Cellar`` directory.
6. For the following commands, *order matters*; items further down the list may depend on previous installs. In addition to this list, double check that all items on `Install from source`_ are also installed.

   .. _MacPorts: http://www.macports.org/install.php

   .. Warning:: If you are using OS-X 10.9, *DO NOT* run ``sudo port install boost``! Instead, you must build boost from source: see `Building Boost`_. If you have installed Boost with MacPorts, then uninstall it. For the curious, `Boost, MacPorts, and OS X 10.9`_ details why this is an issue.

   ::

      $ sudo port selfupdate
      $ sudo port install gcc47
      $ sudo port select --set gcc mp-gcc47
      $ sudo port install cmake
      $ sudo port install python27
      $ sudo port select --set python python27
      $ sudo port install py27-pip
      $ sudo port select --set pip pip27
      $ sudo port install boost  ### <------ DO NOT run this in OS X 10.9!
      $ sudo port install xorg-server
      $ sudo port install py-matplotlib
      $ sudo port install doxygen
      $ export MOE_CMAKE_OPTS=-DCMAKE_FIND_ROOT_PATH=/opt/local && export MOE_CC_PATH=/opt/local/bin/gcc && export MOE_CXX_PATH=/opt/local/bin/g++

   The previous assumes that you want to use ``gcc 4.7`` and ``Python 2.7``; modify the ``install`` and ``set`` invocations if you want other versions.

7. Using ``port select --list``, check that the active versions of gcc, python, etc. are correct. In particular, OS X users want to see ``python27 (active)``, not ``python27-apple (active)``. See `port select information`_.
8. Continue with the installation instructions. If you are having strange errors (no current thread, segfault, etc.), check `Python Tips`_.

General MacPorts Tips
^^^^^^^^^^^^^^^^^^^^^

The `MacPorts Guide`_ provides a detailed introduction to all of MacPorts' features; we will provide a brief overview here.

.. _MacPorts Guide: https://guide.macports.org/

1. ``port install`` and ``port uninstall`` are pretty self-explanatory, being already demonstrated in `OSX Tips`_.
2. ``port selfupdate`` updates MacPorts. MacPorts will warn you when it is out of date.
3. ``port upgrade outdated`` upgrades outdated ports. ``port outdated`` will show you which ports are outdated.
4. ``port list NAME`` lists all ports available for a name. ``port installed NAME`` lists all installed ports with that name.  ``NAME`` can be a regular expression.  For example,

   ::

      $ port installed "boost*"
      yields something like:
      boost @1.51.0_1+no_single+no_static+python27
      boost @1.55.0_2+no_single+no_static+python27 (active)
      boost-build @2.0-m12_2 (active)
      boost-jam @3.1.18_0 (active)

   showing all ports related to Boost. As another example, ``port list "gcc*"`` will show you all ports available related to gcc. These are useful for checking how MacPorts names a particular port, what ports are on your system, and what ports are active.

   .. _port select information:

5. ``port select --list NAME`` will show you available versions of some versioned software managed by MacPorts (e.g., gcc, python, pip). You can change the active version of ``NAME`` by: ``port select --list NAME desired-NAME-version`` where ``desired-foo-version`` is displayed in ``port select --list NAME``.

Boost, MacPorts, and OS X 10.9
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We apologize for the extra complexity around Boost and OS X 10.9. To our knowledge, building Boost from source (`Building Boost`_) is the best solution.

Under OS X 10.9, Apple switched their canonical C++ library from ``libstdc++`` (GNU) to ``libc++`` (LLVM); they are not ABI-compatible. To remain consistent, package managers are linking against ``libc++``. Since MOE is built with gcc, we need ``libstdc++``; thus dependencies must also be built with that C++ library. Currently, package managers do not have enough flexibility to operate several C++ libraries at once, and we do not expect this to change. Ignoring this condition leads to binary incompatibilities; e.g., see:
http://stackoverflow.com/questions/20134223/building-a-boost-python-application-on-macos-10-9-mavericks/

Building Boost
--------------

1. Download the `Boost source`_.
2. From within the main directory, run (after checking additional options below):

   .. _Boost source: http://sourceforge.net/projects/boost/files/boost/1.55.0/

   ::

      $ sudo ./bootstrap.sh --with-python=PYTHON
      $ sudo ./b2 install

   where ``PYTHON`` is the path to your python executable. If you have been following along in OS X, this is ``/opt/local/bin/python``.

3. Make sure ``which gcc`` is ``/opt/local/bin/gcc`` (macport installed) or whatever C++11 compliant gcc you want (similarly, ``which g++`` should be ``/opt/local/bin/g++``), and make sure ``python`` is correct (e.g., ``/opt/local/bin/python`` if using MacPorts).

Additional options for ``./boostrap.sh``:

* ``--with-libraries=python,math,random,program_options,exception,system`` compiles only the libraries we need.
* ``--prefix=path/to/install/dir`` builds Boost and pulls the libraries in the specified path. Default is ``/usr/local`` (recommended, especially if you already have system Boost installations; remember to set ``BOOST_ROOT``).

Additional options for ``./b2``: 

* ``--build-dir=/path/to/build/dir`` builds the Boost files in a separate location instead of mixed into the source tree (recommended).
* ``-j4`` uses 4 threads to compile (faster).

Connecting Boost to MOE
^^^^^^^^^^^^^^^^^^^^^^^

If cmake is unable to find Boost, finds the wrong version of Boost, etc. then try the following:

0. How to specify the ``BOOST_ROOT`` variable: this variable should point to where Boost is installed (e.g., ``/usr/local``). In particular, ``libboost_.*[.a|.so|.dylib]`` files should live in ``${BOOST_ROOT}/lib`` or ``${BOOST_ROOT}/stage/lib`` and boost header files (e.g., ``python.hpp``) should live in ``${BOOST_ROOT}/boost`` or ``${BOOST_ROOT}/include/boost``.
1. When building MOE, add the ``BOOST_ROOT`` variable (described above) to ``MOE_CMAKE_OPTS``. Verify that CMake finds the correct Boost (e.g., in ``moe/build/CMakeCache.txt``, check that the variables ``Boost_INCLUDE_DIR`` and ``Boost_LIBRARY_DIR`` point to your Boost).
2. You might also need to prepend ``BOOST_ROOT`` to ``CMAKE_FIND_ROOT_PATH`` to make this work if you have separate Boost installation(s). For example:

   ::

      $ export MOE_CMAKE_OPTS='-D BOOST_ROOT=/path/to/boost -D Boost_NO_SYSTEM_PATHS=ON -D CMAKE_FIND_ROOT_PATH=/path/to/boost:/opt/local -D OTHER_OPTIONS...'

   ``/opt/local`` is for MacPorts users; it is not needed in Linux and users of other OS X package managers should change this path accordingly.

3. If you elected to use a different Python than the one from MacPorts or are encountering any strange problems, check `Python Tips`_ for how to manually specify Python.

Linux Tips
----------

1. For Ubuntu 13.04+ can apt-get everything you need. Yay for real package managers!

   ::

      $ apt-get update
      $ apt-get install python python-dev gcc cmake libboost-all-dev python-pip doxygen libblas-dev liblapack-dev gfortran git python-numpy python-scipy
      $ pip install -r requirements.txt
      $ python setup.py install
      $ pserve --reload development.ini # MOE server is now running at http://localhost:6543

2. If you are having strange errors (no current thread, segfault, etc.) or need to specify different versions of software (Boost, Python, etc.), check `Python Tips`_ and/or `Connecting Boost to MOE`_.

Ubuntu 12.04 Tips
^^^^^^^^^^^^^^^^^

Ubuntu 12.04 repositories don't contain the versions of ``gcc``, ``cmake``, ``python-numpy`` or ``libboost`` that MOE requires so we need to do some PPA magic::

    # PPA for gcc and g++ 4.7
    $ sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
    # PPA for boost 1.55
    $ sudo add-apt-repository -y ppa:boost-latest/ppa
    # PPA for cmake 1.8.12.2
    $ sudo add-apt-repository -y ppa:kalakris/cmake
    # PPA for numpy 1.8.1
    $ sudo add-apt-repository -y ppa:chris-lea/python-numpy
    $ sudo apt-get update -qq
    $ sudo apt-get install -y build-essential python python-dev python2.7 python2.7-dev doxygen libblas-dev liblapack-dev gfortran git make flex bison libssl-dev libedit-dev python-scipy gcc-4.7 g++-4.7 boost1.55 cmake python-numpy
    # Now we need to tell ubuntu to use the correct gcc/g++
    $ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.7 20
    $ sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.7 20
    $ sudo update-alternatives --config gcc
    $ sudo update-alternatives --config g++
    $ pip install -r requirements.txt
    $ python setup.py install

If you are having strange errors (no current thread, segfault, etc.) or need to specify different versions of software (Boost, Python, etc.), check `Python Tips`_ and/or `Connecting Boost to MOE`_.

CMake Tips
----------

1. Do you have dependencies installed in non-standard places? e.g., did you build your own boost? Set the env var: ``export MOE_CMAKE_OPTS=-DCMAKE_FIND_ROOT_PATH=/path/to/your/dependencies ...`` (OS X users with MacPorts should set ``/opt/local``.) This can be used to set any number of cmake arguments.
2. Have you checked `Connecting Boost to MOE`_ and `Python Tips`_?
3. Are you using the right compiler? e.g., for ``gcc``, run ``export MOE_CC_PATH=/path/to/your/gcc && export MOE_CXX_PATH=/path/to/your/g++`` (OS X users need to explicitly set this.)

Python Tips
-----------

.. Note:: This is an advanced-user section. ``setup.py`` should be able to identify the correct Python automatically (i.e., it tries to find the Python it was launched with). Examples of why you might need to keep reading: 1) ``setup.py`` failed to find the correct Python paths; 2) you building manually and not using ``setup.py``; 3) you are doing something "weird" like building MOE with a different version of Python than the one you intend to run MOE with.

Sometimes cmake and/or ``setup.py`` will fail to find your Python installation or you will want to specify an alternate Python. To specify Python, add:

::

   -D MOE_PYTHON_INCLUDE_DIR=/path/to/where/Python.h/is/found
   -D MOE_PYTHON_LIBRARY=/path/to/python/shared/library/object

to the ``MOE_CMAKE_OPTS`` environment variable. Note that options added to this environment variable *supersede* options set by ``setup.py``; so if ``setup.py`` failed, manually specifying the right paths will solve the problem. For example, an OS X user might have:

::

   export MOE_CMAKE_OPTS='-D CMAKE_FIND_ROOT_PATH=/opt/local -D MOE_PYTHON_INCLUDE_DIR=/System/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7/ -D MOE_PYTHON_LIBRARY=/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/config/libpython2.7.dylib'

In OS X, the python dynamic library will be a ``.dylib`` file; in Linux, it will be a ``.so`` file.

.. WARNING:: Mis-matched Python versions between your virtual environment, Boost, and/or MOE's installer can lead to a plethora of strange bugs. Anywhere from ``Fatal Python error: PyThreadState_Get: no current thread`` to segmentation faults and beyond. (You are hitting a binary incompatibility so it is hard to predict the specific error.)  You may need to instruct your package manager to build Boost against a particular version of Python, indicate a different Python to MOE, etc. to make these versions line up.

Here are some ways to check/ensure that Python was found and linked correctly:

1. You can verify that cmake found the correct version by checking the values of ``PYTHON_INCLUDE_DIR`` and ``PYTHON_LIBRARY`` in ``moe/build/CMakeCache.txt``.
2. In `General MacPorts Tips`_, *notice* that Boost is built against ``python27``. Checking ``port installed "python*"``, you should see (amongst others) ``python27 @2.7.6_0 (active)``.
3. ``python --version`` will show you what version of Python is called by default.
4. Outside of a virtual environment, running ``which python`` (and tracking through the symlinks; the first level should be in ``/opt/local/...`` if you are using MacPorts in OS X) will show you specifically which Python is being used.
5. Inside of a virtual environment, ``yolk -l`` will show you what software versions are in use. The path to Python should match the Python used to install Boost and MOE. (Running ``which python`` still works here if you trace through the symlinks.) Get ``yolk`` via ``pip install yolk``.
6. Check binary shared library dependencies (only works if you are not linking statically). ``locate libboost_python`` and run ``ldd`` (Linux) or ``otool -L`` (OS X) on the dynamic library.  (Note: ``ldd`` in Linux may not show the Python dependency since this linkage may be delayed till actual use.)  Similarly, running those commands on ``moe/build/GPP.so`` should show you the same Python as above; for example:

   ::

      LINUX:
      $ ldd moe/build/GPP.so
      yields lines like:
      libpython2.7.so.1.0 => /usr/lib/libpython2.7.so.1.0 (0x00007f7d7a9fc000)

      OS X:
      $ otool -L moe/build/GPP.so
      yields:
      /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python (compatibility version 2.7.0, current version 2.7.0)

   This should be the same Python that you see in the other steps.

   If you linked statically, you need to check your link lines manually. Since MOE links dynamically by default, we assume that you know what you are doing if you changed it.
