Contributing
************

Anyone and everyone is encouraged to help work on and develop for MOE. Below are some tips and tricks to help you along.

**Contents:**

    #. `Making a pull request`_
    #. `Documentation`_
    #. `Testing`_
    #. `Style`_
    #. `Versioning`_
    #. `Releasing (For Maintainers)`_

Making a pull request
---------------------

1. Fork it.
2. Create a branch (``git checkout -b my_moe_branch``)
3. Develop your feature/fix (don't forget to add tests and docs!)
4. Run tests (``make test``)
5. Build docs (``make docs``)
6. Test against styleguide (``make style-test``)
7. Commit your changes (``git commit -am "Added Some Mathemagics"``)
8. Push to the branch (``git push origin my_moe_branch``)
9. Update the CHANGELOG (`CHANGELOG Updates`_)
10. Open a Pull Request - http://github.com/Yelp/MOE/pulls (Use the template here: `Pull request review templates`_)
11. Optimize locally while you wait

CHANGELOG Updates
.................

Before your change will be accepted into master, you must update ``CHANGELOG.md`` in the MOE root directory. The changelog contains a listing of major changes associated with each release, broken down into relevant categories (e.g., ``Features``, ``Changes``, ``Bugs``). The top of the changelog is for the current MOE version under development; this is where your changes go. Past releases are listed below the current candidate in reverse-chronological order.

Add your changes under the appropriate categories at the TOP of ``CHANGELOG.md``. Add new categories as needed.

The changelog might look like:

::

   * Features
     * Cool new feature
     * Even cooler, newer feature
     * YOUR NEW FEATURE HERE!
   * Changes
     * Earth-shattering change
   * Bugs
     * Running MOE no longer causes black holes

Pull request review templates
.............................

::

    ********* PEOPLE *************
    Primary reviewer:

    Reviewers: 

    ********* DESCRIPTION **************
    Branch Name:
    Ticket(s)/Issue(s): Closes #XX

    ********* TESTING DONE *************

``PEOPLE`` section
^^^^^^^^^^^^^^^^^^

``Primary reviewer``: the primary code reviewer. This person is *EQUALLY RESPONSIBLE* for your branch. They should read and familiarize themselves with all aspects of the code, checking for style, correctness, maintainability, testing, etc. If you do not know who to put then use @sc932 and/or @suntzu86.

``Reviewers``: the people (including primary) that you would like to read your branch. Please use full email addresses or @<GITHUB NAME>.

``DESCRIPTION`` section
^^^^^^^^^^^^^^^^^^^^^^^

``Branch Name``: name of your branch on git. we use the style::

  username_ticket#_some_brief_description_of_stuff

Examples::

  suntzu86_ads_3094_python_implementation_of_ABCs
  suntzu86_gh_74_fix_todos_ticket_numbers_comments
  sc932_21_fix_tooltips

``Ticket(s)/Issue(s)``: usually GH-##. Github will autolink “GH-##” ticket numbers. If your tickets live elsewhere (e.g., JIRA), provide a link. You can also do #XX

Description of the changes goes here.

You should highlight major features of this branch and the merge. Is there junk from pulling master? Do specific files/pieces need extra attention? Do readers need any extra background to understand what you’re doing?
Tell us about your work!

.. Note::
    
    Don't feel the need to copy/paste the ticket

``TESTING DONE`` section
^^^^^^^^^^^^^^^^^^^^^^^^

What testing have you done? You should AT LEAST have compiled your code (if appplicable, e.g., C++ changes) and run::

  make test
  make style-test
  make docs

If your changes are uncovered by tests:

1. why? can you add a test?
2. if not, at least test the changes ad-hoc. e.g., see that the new text/button shows up on the UI, force your code to execute and see that things “look right”, etc. *EXPLAIN AND JUSTIFY* yourself if you go this route. Your reviewers will hopefully have suggestions for how to turn scenario 2) into 1).

Documentation
-------------

Documentation is a very important component for MOE. The complex math and multiple languages neccesitate clearly documented APIs and explainations. All new code needs to meet the documentations standards.

Building the documentation
..........................

First check it locally (``make docs``), the built docs will be in <MOE_DIR>/docs/_build/html/index.html.

To update the online documentation::

    git checkout gh-pages
    git pull origin master
    rm -r docs/_build/html
    make docs
    cp -r docs/_build/html/* .
    git add -A
    git commit -m "updated online docs" --no-verify
    git push origin HEAD

Python Documentation
....................

MOE follows the pep257 (http://legacy.python.org/dev/peps/pep-0257) conventions for docstrings and (most of) ``pep8`` for style (http://legacy.python.org/dev/peps/pep-0008). These conventions are inforced using the ``flake8`` docstrings module (run using ``make style-test``).

.. Note::

    All new python code must follow the ``pep257`` docstring conventions and ``pep8`` style conventions.

All documentation is built using the ``sphinx-apidoc`` command. For more information see http://sphinx-doc.org/man/sphinx-apidoc.html. Support for :math:`\LaTeX` is also included.

C++ Documentation
.................

MOE uses ``doxygen`` (http://www.stack.nl/~dimitri/doxygen) to extract the C++ documentation from the source. An API is then generated in ``sphinx`` through ``breathe`` (http://breathe.readthedocs.org/en/latest). All sphinx ReStructured Text markup is available and should be used when writing new C++ code.

Testing
-------

MOE currently uses ``testify`` (https://github.com/Yelp/Testify) to run all unit and integration tests.

Tests can be run using ``make test``. Continuous integration testing is provided by http://travis-ci.org. All builds are tested in a fresh ubuntu VM and need to be passing before being pulled into master.

.. Note::

    All new code should be tested before submitting a pull request.

Documentation for and examples of tests can be found at :doc:`moe.tests`

Style
-----

MOE uses the google style guides found here:

* python - http://google-styleguide.googlecode.com/svn/trunk/pyguide.html
* C++ - http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml

.. Note::

    All new code should conform to these style guides

Versioning
----------

MOE uses semantic versioning (http://semver.org). To summarize, our patch numbers look like:

::

   MAJOR.MINOR.PATCH

1. ``MAJOR``: incremented for incompatible API changes (this includes the REST, Python, and C++ interfaces; e.g., removing functionality from an endpoint, modifying arguments to a library call)
2. ``MINOR``: incremented for adding functionality in a backwards-compatible manner (e.g., adding a new REST endpoint, adding capabilities to an existing endpoint)
3. ``PATCH``: incremented for backward-compatible bug fixes and minor capability improvements (e.g., fixing bugs, performance improvements, code cleanup/refactoring)

We do not increment versions for documentation and other non-code changes.

Releasing (For Maintainers)
---------------------------

The MOE repository maintainers decide when to tag official releases. They may decide to bump versions immediately after a pull request for a critical bug fix, or they may decide to wait and combine several inbound pull requests into one version bump. (Having ``MOE v10.87.3091`` makes the history unwieldy).

Tagging Releases
................

#. On the GitHub releases page (https://github.com/Yelp/MOE/releases), click ``Draft a new release``.
#. Choose a new version number; see `Versioning`_.
#. Select a target branch (most likely ``master``).
#. Name the release (``Release Title`` field) with just the version number.
#. In the description field, copy-paste the current release decriptors from ``CHANGELOG.md``.
#. Update ``CHANGELOG.md``. Label the current set of features, changes, etc. with the new version number and date in parenthesis, preceded by a ``##`` header marker. Write down the associated SHA. And make new category headers at the top of the file for future commits. For example, after marking the current version for release, change ``CHANGELOG.md`` to look like:

   ::

      * Features

      * Changes

      * Bugs

      ## vMAJOR.MINOR.PATCH (YEAR-MONTH-DAY)

      SHA: ``271828someshagoeshere314159``

      * Features
        * Copied from ``CHANGELOG.md``
        * More features!
      * Changes
        * etc

Updating DockerHub
..................

TODO(sclark)
