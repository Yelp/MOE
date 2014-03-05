import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(here, 'README.md')).read()

requires = [
    'pyramid',
    'WebError',
    'pymongo',
    'testify',
    'webtest',
    'nose',
    'yolk',
    'numpy',
    'scipy',
    'matplotlib',
    'simplejson',
    'colander',
    ]

setup(name='MOE',
      version='0.0.1',
      description='Metric Optimization Engine',
      long_description=README,
      classifiers=[
        "Programming Language :: Python",
        "Framework :: Pylons",
        "Topic :: Internet :: WWW/HTTP",
        "Topic :: Internet :: WWW/HTTP :: WSGI :: Application",
        ],
      author="Scott Clark and Eric Liu",
      author_email='sclark@yelp.com',
      url='https://github.com/sc932/MOE',
      keywords='web pyramid pylons mongodb',
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
      )

