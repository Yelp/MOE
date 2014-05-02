# -*- coding: utf-8 -*-
"""Setup for the MOE webapp."""
import os
import shutil
import subprocess
from collections import namedtuple

from setuptools import setup, find_packages
from setuptools.command.install import install

here = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(here, 'README.md')).read()

# Following the versioning system at http://semver.org/
MAJOR = 0
MINOR = 1
MICRO = 0
VERSION = "%d.%d.%d" % (MAJOR, MINOR, MICRO)

CLASSIFIERS = """
        Development Status :: 4 - Beta
        Intended Audience :: Science/Research
        Intended Audience :: Developers
        Programming Language :: C++
        Programming Language :: Python
        Topic :: Software Development
        Topic :: Scientific/Engineering
        Operating System :: Unix
        Operating System :: MacOS

        """

# If you change something here, change it in requirements.txt
requires = [
    'pyramid',
    'pyramid_mako',
    'WebError',
    'testify',
    'webtest',
    'tox',
    'numpy',
    'scipy',
    'simplejson',
    'colander',
    'sphinx',
    'breathe',
    'sphinxcontrib-httpdomain',
    'sphinx_rtd_theme',
    ]

MoeExecutable = namedtuple('MoeExecutable', ['env_var', 'exe_name'])


def find_path(moe_executable):
    """Return the path for an executable, or None if it cannot be found.

    Performs the search in the following way:
    1. Check the env var MOE_<EXECUTABLE>
    2. Tries to find the executable in the system $PATH

    """
    # First we see if the env var is set
    path = os.environ.get(moe_executable.env_var, None)
    if path is not None:
        return path

    # Try to guess where the executable is
    for prefix in os.environ.get("PATH").split(os.pathsep):
        potential_path = os.path.join(prefix, moe_executable.exe_name)
        if os.path.isfile(potential_path):
            path = potential_path
            print "Could not find env var %s for %s, using %s from $PATH" % (moe_executable.env_var, moe_executable.exe_name, path)
            break

    return path


class InstallCppComponents(install):

    """Install required C++ components."""

    def run(self):
        """Run the install."""
        install.run(self)
        
        # Sometimes we want to manually build the C++ (like in Docker)
        if os.environ.get('MOE_NO_BUILD_CPP', 'False') == 'True':
            return

        package_dir = os.path.join(self.install_lib, 'moe')
        build_dir = os.path.join(package_dir, 'build')

        cmake_path = find_path(
                MoeExecutable(
                    env_var='MOE_CMAKE_PATH',
                    exe_name='cmake',
                    )
                )

        cmake_options = os.environ.get('MOE_CMAKE_OPTS', '')
        if cmake_options == '':
            print "MOE_CMAKE_OPTS not set. Passing no extra args to cmake."
        else:
            print "Passing '%s' args from MOE_CMAKE_OPTS to cmake." % cmake_options

        # Set env dict with cc and/or cxx path
        cc_path = os.environ.get('MOE_CC_PATH', None)
        cxx_path = os.environ.get('MOE_CXX_PATH', None)
        env = {}
        if cc_path is not None:
            env['CC'] = cc_path
            print "Passing CC=%s to cmake." % (cc_path)
        else:
            print "MOE_CC_PATH not set. Not passing a CC env var to cmake."
        if cxx_path is not None:
            env['CXX'] = cxx_path
            print "Passing CXX=%s to cmake." % (cxx_path)
        else:
            print "MOE_CXX_PATH not set. Not passing a CXX env var to cmake."

        # Set cmake if not already set

        # rm the build directory if it exists
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir)
        os.mkdir(build_dir)
        local_build_dir = os.path.join(here, 'moe', 'build')
        if os.path.exists(local_build_dir):
            shutil.rmtree(local_build_dir)
        os.mkdir(local_build_dir)

        cpp_location = os.path.join(here, 'moe', 'optimal_learning', 'cpp')

        # Run cmake
        proc = subprocess.Popen(
                [
                    cmake_path,
                    cmake_options,
                    cpp_location,
                    ],
                cwd=local_build_dir,
                env=env,
                )
        proc.wait()

        # Compile everything
        proc = subprocess.Popen(["make"], cwd=local_build_dir, env=env)
        proc.wait()

        GPP_so = os.path.join(local_build_dir, 'GPP.so')
        build_init = os.path.join(local_build_dir, '__init__.py')

        shutil.copyfile(GPP_so, os.path.join(build_dir, 'GPP.so'))
        shutil.copyfile(build_init, os.path.join(build_dir, '__init__.py'))


setup(name='MOE',
      version=VERSION,
      description='Metric Optimization Engine',
      long_description=README,
      classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
      author="Scott Clark and Eric Liu",
      author_email='sclark@yelp.com',
      url='https://github.com/sc932/MOE',
      keywords='optimization optimal learning EI experiment design',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=requires,
      tests_require=requires,
      test_suite="moe",
      entry_points = """\
      [paste.app_factory]
      main = moe:main
      """,
      paster_plugins=['pyramid'],
      cmdclass={
          'install': InstallCppComponents,
          },
      )
