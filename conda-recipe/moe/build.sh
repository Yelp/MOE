#!/bin/bash
export CMAKE_C_COMPILER=/usr/bin/gcc
export CMAKE_C_COMPILER=/usr/bin/g++-4.7

$PYTHON setup.py install

# Add more build steps here, if they are necessary.

# See
# http://docs.continuum.io/conda/build.html
# for a list of environment variables that are set during the build process.
