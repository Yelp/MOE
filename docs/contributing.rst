Contributing
*******

Anyone and everyone is encouraged to help work on and develop for MOE. Below are some tips and tricks to help you along.

**Contents:**

    #. `Making a pull request`_
    #. `Documentation`_
    #. `Testing`_

Making a pull request
-----
1. Fork it.
2. Create a branch (``git checkout -b my_moe_branch``)
3. Develop your feature/fix (don't forget to add tests and docs!)
4. Run tests (``tox`` or ``make test-no-tox``)
5. Build docs (``make docs-no-tox``)
6. Test against styleguide (``tox -e pep8`` or ``make style-test-no-tox``)
7. Commit your changes (``git commit -am "Added Some Mathemagics"``)
8. Push to the branch (``git push origin my_moe_branch``)
9. Open a Pull Request - http://github.com/sc932/MOE/pulls
10. Optimize locally while you wait

Pull request review templates
....

::

    ********* PEOPLE *************
    Primary reviewer:

    Reviewers: 

    ********* DESCRIPTION **************
    Branch Name:
    Ticket(s)/Issue(s): Closes #XX

    ********* TESTING DONE *************

``PEOPLE`` section
^^^^^

``Primary reviewer``: the primary code reviewer. This person is *EQUALLY RESPONSIBLE* for your branch. They should read and familiarize themselves with all aspects of the code, checking for style, correctness, maintainability, testing, etc.
Ask the primary reviewer first if your branch is particularly large (and try to avoid large branches, anything over 300+ lines).

``Reviewers``: the people (including primary) that you would like to read your branch. Please use full email addresses or @<GITHUB NAME>. Right now, only people shown here: https://github.com/sc932/MOE/watchers can see the review.

``DESCRIPTION`` section
^^^^^
``Branch Name``: name of your branch on git. we use the style::

  username_ticket#_some_brief_description_of_stuff

Examples::

  eliu_ads_3094_python_implementation_of_ABCs
  eliu_gh_74_fix_todos_ticket_numbers_comments
  sclark_21_fix_tooltips

``Ticket(s)/Issue(s)``: usually GH-##. Github will autolink “GH-##” ticket numbers. If your tickets live elsewhere (e.g., JIRA), provide a link. You can also do #XX

Description of the changes goes here.

You should highlight major features of this branch and the merge. Is there junk from pulling master? Do specific files/pieces need extra attention? Do readers need any extra background to understand what you’re doing?
Tell us about your work!

.. Note::
    
    Don't feel the need to copy/paste the ticket

``TESTING DONE`` section
^^^^^^^
What testing have you done? You should AT LEAST have compiled your code (if appplicable, e.g., C++ changes) and run::

  make test-no-tox
  make style-test-no-tox

If your changes are uncovered by tests:

1. why? can you add a test?
2. if not, at least test the changes ad-hoc. e.g., see that the new text/button shows up on the UI, force your code to execute and see that things “look right”, etc. *EXPLAIN AND JUSTIFY* yourself if you go this route. Your reviewers will hopefully have suggestions for how to turn scenario 2) into 1).

Documentation
-----

Documentation is a very important component for MOE. The complex math and multiple languages neccesitate clearly documented APIs and explainations. All new code needs to meet the documentations standards.

Building the documentation
......

First check it locally (``make docs-no-tox``), the built docs will be in <MOE_DIR>/docs/_build/html/index.html.

To update the online documentation::

    git checkout gh-pages
    git pull origin master
    make docs-no-tox
    cp -r docs/_build/html/* .
    git add -A
    git commit -m "updated online docs" --no-verify
    git push origin HEAD

Python Documentation
....

MOE follows the pep257 (http://legacy.python.org/dev/peps/pep-0257) conventions for docstrings and (most of) ``pep8`` for style (http://legacy.python.org/dev/peps/pep-0008). These conventions are inforced using the ``flake8`` docstrings module (run using ``make style-test-no-tox``).

.. Note::

    All new python code must follow the ``pep257`` docstring conventions and ``pep8`` style conventions.

All documentation is built using the ``sphinx-apidoc`` command. For more information see http://sphinx-doc.org/man/sphinx-apidoc.html. Support for :math:`\LaTeX` is also included.

C++ Documentation
.....

MOE uses ``doxygen`` (http://www.stack.nl/~dimitri/doxygen) to extract the C++ documentation from the source. An API is then generated in ``sphinx`` through ``breathe`` (http://breathe.readthedocs.org/en/latest). All sphinx ReStructured Text markup is available and should be used when writing new C++ code.

Testing
-----

MOE currently uses ``testify`` (https://github.com/Yelp/Testify) to run all unit and integration tests.

.. Note::

    All new code should be tested before submitting a pull request.

Documentation for and examples of tests can be found at :doc:`moe.tests`
