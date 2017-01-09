* Features

* Changes

* Bugs

## v0.2.2 (2014-12-5)

SHA: ``83a71512e6a4aa553866e4301ef11f172d485e23``

* Features

  * Added startup message to REST server including tips for OSX users (#400)
  * Added GPU support to ``cpp_wrappers.expected_improvement.multistart_expected_improvement_optimization``; requires ``max_num_threads == 1`` until future multi-GPU support (#368)
  * Added the COBYLA optimizer to the expected improvement optimization class. (#370)
  * OptimizerParameter struct members now directly readable/writeable from Python; added EqualityComparisonMixin (#138)
  * C++ GradientDescentParameters object now stores ``num_steps_averaged`` (but does not use it yet) (#391)
  * conda build system for MOE (#417)

* Changes

  * Switched from `testify` to `py.test` - http://pytest.org/ (#36)
  * [cleanup] Moved bandits into their own sub directories (#375)
  * Supply PYTHON_LIBRARY and PYTHON_INCLUDE_DIR vars to cmake automatically in ``setup.py`` (#412)
  * Added warning when colander version is out of date (#413)

* Bugs

  * Fixed UCB1 and UCB1-tuned algorithm confidence bound calculation (#432)

## v0.2.1 (2014-09-22)

SHA: ``ab6f959c11a0cacbed6dad618fe6ffed71092116``

* Features

  * Implemented BLA (Bayesian Learning Automaton). (#373)
  * Connected GPU functions to multistart gradient descent optimizer. (#270)

* Changes

* Bugs

  * variance in a sample arm was dropped in _make_bandit_historical_info_from_params. (#385)
  * SampleArm's ``__add__`` and ``__str__`` were broken. (#387)
  * Specifying ``max_num_threads`` on GPU compute paths caused a segfault (#394)

## v0.2.0 (2014-08-15)

SHA: ``8201917e3f9b47b8edd8039ea3278ef8631b0f2a``

* Features

  * Added multi-armed bandit endpoint. (#255)
    * Implemented epsilon-greedy. (#255)
    * Implemented epsilon-first. (#335)
    * Implemented UCB1. (#354)
    * Implemented UCB1-tuned. (#366)
  * Added support for the L-BFGS-B optimizer. (#296)
  * Added GPU implementation for q,p-EI and its gradient computation. (#219)
    * Speed up GPU functions by redesign of memory allocation. (#297)

* Changes

  * Split up old ``schemas.py`` file into ``schemas/`` directory with several subfiles (#291)
  * Improved Dockerfile, reducing Docker-based install times substantially, https://hub.docker.com/u/yelpmoe/ (#332)
    * Created ``min_reqs`` docker container which is a snapshot of all MOE third-party requirements
    * Created ``latest``, which tracks the latest MOE build
    * Started releasing docker containers for each tagged MOE release (currently just ``v0.1.0``)
  * ``GradientDescentOptimization`` (C++) no longer has a separate ``next_points`` output (#186)
  * LogLikelihood evaluate at point list and latin hypercube search now return status dicts like every other optimizer (#189)
    * status dicts also a little more informative/standardized now
  * Update C++ autodoc tools to handle the new ``gpu`` directory (#353)
  * Added ``__version__`` to ``moe/__init__.py`` (#353)

* Bugs

  * Throw exceptions (C++) if ``num_multistarts`` or ``num_random_samples`` is 0 (#345)
  * ``combined_example`` endpoint was not passing ``kwargs`` through so users could not change the default server (#356)
    * fix sometimes dropped general ``kwargs`` (#358)
  * ``mean_var_of_gp_from_historic_data`` was also not passing ``kwargs`` (#359)

## v0.1.0 (2014-07-29)

SHA: ``5fef1d242cc8b6e0d6443522f8ba73ba743607de``

* Features

  * initial open source release
