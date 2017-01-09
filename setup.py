# -*- coding: utf-8 -*-
"""Setup for the MOE webapp."""
import os
import shlex
import shutil
import subprocess
import sys
import warnings
from collections import namedtuple

from moe import __version__

from setuptools import setup, find_packages
from setuptools.command.install import install

try:
    import sysconfig
except ImportError:
    from distutils import sysconfig


here = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(here, 'README.md')).read()


VERSION = __version__


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
    'pytest',
    'webtest',
    'simplejson',
    'numpy',
    'scipy',
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
            print "Could not find env var {0:s} for {1:s}, using {2:s} from $PATH".format(moe_executable.env_var, moe_executable.exe_name, path)
            break

    return path


class InstallCppComponents(install):

    """Install required C++ components."""

    def run(self):
        """Run the install."""
        install.run(self)
        # Copy so we do not overwrite the user's environment later.
        env = os.environ.copy()

        # Sometimes we want to manually build the C++ (like in Docker)
        if env.get('MOE_NO_BUILD_CPP', 'False') == 'True':
            return

        package_dir = os.path.join(self.install_lib, 'moe')
        build_dir = os.path.join(package_dir, 'build')

        cmake_path = find_path(
            MoeExecutable(
                env_var='MOE_CMAKE_PATH',
                exe_name='cmake',
                )
            )

        cmake_options = env.get('MOE_CMAKE_OPTS', '')
        if cmake_options == '':
            print "MOE_CMAKE_OPTS not set. Passing no extra args to cmake."
        else:
            print "Passing '{0:s}' args from MOE_CMAKE_OPTS to cmake.".format(cmake_options)

        # Set env dict with cc and/or cxx path
        # To find C/C++ compilers, we first try read MOE_CC/CXX_PATH and if they exist, we write to CC/CXX.
        # Then we read and pass CC/CXX to cmake if they are set.
        if 'MOE_CC_PATH' in env:
            env['CC'] = env['MOE_CC_PATH']
        if 'MOE_CXX_PATH' in env:
            env['CXX'] = env['MOE_CXX_PATH']
        if 'CC' in env:
            print "Passing CC={0:s} to cmake.".format(env['CC'])
        if 'CXX' in env:
            print "Passing CXX={0:s} to cmake.".format(env['CXX'])

        # rm the build directory if it exists
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir)
        os.mkdir(build_dir)
        local_build_dir = os.path.join(here, 'moe', 'build')
        if os.path.exists(local_build_dir):
            shutil.rmtree(local_build_dir)
        os.mkdir(local_build_dir)

        cpp_location = os.path.join(here, 'moe', 'optimal_learning', 'cpp')

        # Reformat options string: options & args that are separated by whitespace on the command line
        # must be passed to subprocess.Popen in separate list elements.
        cmake_options_split = shlex.split(cmake_options)

        # Get info on Python paths (for the currently running Python) that we need to discover
        # header/library locations for MOE
        includepy, libdir, instsoname = sysconfig.get_config_vars('INCLUDEPY', 'LIBDIR', 'INSTSONAME')

        # The meaning of 'LIBDIR' and 'INSTSONAME' is platform-specific. Handle the various cases.
        # Initial value: 'LIBDIR' usually contains what we want, but one OSX case requires a different path.
        moe_python_library_base = libdir
        if not sys.platform.startswith('darwin'):
            # For other platforms (only Linux is tested!) libdir + instsoname yields the shared object location.
            if not sys.platform.startswith('linux'):
                warnings.warn("Automatic path discovery untested outside of OSX and Linux. Taking our best guess.\n"
                              "Please check PYTHON_INCLUDE_DIR and PYTHON_LIBRARY paths below.")
        else:
            # on OSX versions that we tested against, LIBDIR looks like:
            # /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib
            # and INSTSONAME looks like:
            # Python.framework/Versions/2.7/Python
            # What we want is:
            # /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python

            # This is easy in Linux b/c INSTSONAME is the name of a dynamic library file
            # in foo/bar/lib whereas in OSX it's a partial path.
            # Instead, remove the overlapping part of both paths to construct the result.
            instsoname_components = instsoname.rsplit('/', 1)
            framework_subpath = instsoname_components[0]  # Probably is similar to 'Python.framework/Versions/2.7'
            libdir_base, libdir_framework_subpath, libdir_subpath = libdir.rpartition(framework_subpath)

            if not libdir_base and not libdir_framework_subpath:
                # Did not find framework_subpath; the expected overlap isn't there so just join libdir + instsoname
                warnings.warn("Unexpected OSX library paths.\n"
                              "Please check PYTHON_INCLUDE_DIR and PYTHON_LIBRARY paths below.")
            else:
                # Found framework_subpath overlap so skip the overlapped components
                moe_python_library_base = libdir_base

        moe_python_include_dir = includepy
        moe_python_library = os.path.join(moe_python_library_base, instsoname)

        # Print the Python paths we found so that the user can verify them if something goes wrong.
        print 'PYTHON_INCLUDE_DIR (Expected full path to where Python.h is found): {0:s}'.format(moe_python_include_dir)
        print 'PYTHON_LIBRARY (Expected path to Python shared object; e.g., libpython2.7.so or .dylib): {0:s}'.format(moe_python_library)

        # Build the full cmake command using properly tokenized options
        cmake_full_command = [
            cmake_path,
            '-DMOE_PYTHON_INCLUDE_DIR=' + moe_python_include_dir,
            '-DMOE_PYTHON_LIBRARY=' + moe_python_library,
            ]
        cmake_full_command.extend(cmake_options_split)
        cmake_full_command.append(cpp_location)

        # Run cmake
        proc = subprocess.Popen(
            cmake_full_command,
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
      author_email='opensource+moe@yelp.com',
      url='https://github.com/Yelp/MOE',
      keywords='bayesian global optimization optimal learning expected improvement experiment design',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=requires,
      tests_require=requires,
      test_suite="moe",
      entry_points="""\
      [paste.app_factory]
      main = moe:main
      """,
      paster_plugins=['pyramid'],
      cmdclass={
          'install': InstallCppComponents,
          },
      )
