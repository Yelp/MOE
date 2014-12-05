conda-moe
=========

This repository provides the recipes for building binary packages of [MOE](https://github.com/yelp/moe), a global, black box optimization engine for real world metric optimization, to be used with the Anaconda Python distribution or other conda environments.

The built binaries are available on binstar.org, and can be installed using the following command

```
conda config --add channels https://conda.binstar.org/rmcgibbo
conda install moe
```

This package is available for linux-64, targetting Ubuntu 10.04 or later, CentOS 6.0+ or later, or similar.

### Notes
This repository obviously includes recipes for a number of packages other than MOE. These are all direct or indirect dependencies of MOE, which are not included with the `default` conda channels, and we therefore build and push to binstar to make things _just work_.

### libstdc++

MOE uses C++11 features, which requires a quite modern version of GCC (4.7.3+), and a similarly new version of the runtime c++ standard library, libstd++. The version of libstdc++ distributed with Ubuntu 10.04 and CentOS 6.0 is
too old to support runtime C++11 code. Therefore, this conda build of MOE depends on an external libstdc++ conda package, which povides a more recent library. See [the recipe](https://github.com/rmcgibbo/conda-moe/blob/master/libstdcplusplus/build.sh) for the dirty, dirty hack.

### building the binaries

Our builds of these conda recipes were performed on a fresh Ubuntu 10.04 64-bit VM (vagrant) with the following provisioning:

```
sudo apt-get update
sudo apt-get install python-software-properties
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install libgcc-4.7-dev  cpp-4.7 gcc-4.7-base  g++-4.7 make libbz2-1.0 libbz2-dev git-core
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.7 20
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.7 20

wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
bash Miniconda-latest-Linux-x86_64.sh -b
export PATH=$HOME/miniconda/bin/:$PATH
conda install conda-build

# The conda recipes for the boost dependency are in a separate github
# repository:
# https://github.com/rmcgibbo/conda-cheminformatics/tree/master/boost
conda config --add channels https://conda.binstar.org/rmcgibbo

git clone https://github.com/rmcgibbo/conda-moe.git
cd conda-moe
conda build *
```
