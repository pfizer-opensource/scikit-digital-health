.. _devindex:

############
Contributing
############

Contributing to ``scikit-digital-health`` is fairly straightforward, and the process is laid out below.  Much of the guidelines borrow from and are very similar to ``NumPy's`` guidelines.

Development process - summary
=============================

This is the short summary, complete descriptions are below:

1. If you are contributing for the first time:

    * Go to https://github.com/PfizerRD/scikit-digital-health and use the "fork" button to create a version that will create a copy for yourself.
    * Clone the project to your local computer::

        git clone https://github.com/your-username/scikit-digital-health.git
    
    * Change the directory::

        cd scikit-digital-health
    
    * Add the upstream repository::

        git remote add upstream https://github.com/PfizerRD/scikit-digital-health.git
    
    * After adding the upstream repository, `git remote -v` will show two remote repositories:

        - ``upstream``, the `scikit-digital-health` repository
        - ``origin``: your personal fork

2. Add your contribution:

    * Pull latest changes from upstream::

        git checkout master
        git pull upstream master
    
    * Create a branch for the feature you want to contribute. *The branch name should be a descriptive message about your contribution*::

        git checkout -b add-new-gait-metrics
    
    * Commit locally often (``git add`` and ``git commit``), using descriptive messages. 
    
    * Your contribution must include tests. Ideally, these tests are written beforehand, and fail before your full contribution has been implemented.

    * Write the documentation for your contributions, and make sure to update any existing docstrings that have changed as well.

    * *Before submitting your contribution, make sure all unit tests pass, and the documentation builds properly*.

3. To submit your contribution:

    * Push your changes back to your fork on GitHub::

        git push origin add-new-gait-metrics
    
    * On GitHub, the new branch will show up with a green Pull Request (PR) button. Make sure the title and message are *clear, concise, and self-explanatory*. Submit the PR.

4. Tentative review process:

    * Reviewers might write comments regarding your PR in order to improve it and make sure the standards are met.

    * To update your PR, make changes on your local repository, and commit. **Run tests again, and only if they succed** push to your fork. As the changes are pushed up (to the same branch as before) the PR will update automatically.

    * If tests are failing, but you are not sure how to fix these, you can push the changes and ask for help in a PR comment.

    * Continuous Integration is used for PRs, and must pass before the PR can be merged. To avoid overuse and waste of the resources, **test your work locally** before committing.

5. Changes file

    * If your contribution is significant (ie more than docstring updates, small fixes) and introduces new features or significant bugfixes, the changes should be documented in a changes file. Create a short summary file that matches your branch name, e.g. ``changes/add-new-gait-metrics.md``

6. Cross referencing issues

    * If the PR relates to any open issues, add the text ``xref gh-xxxx`` where ``xxxx`` is the number of the issue to GitHub comments. 

    * If the PR instead solves any open issues, add the text ``closes gh-xxxx`` or ``fixes`` or any of the other acceptable key-words `github accepts <https://help.github.com/en/articles/closing-issues-using-keywords>`_.

.. _guidelines:

Guidelines
----------

* All code should have tests.
* All code should be documented.
* Changes require review before merging.

Stylistic Guidelines
--------------------

* Make sure you are following `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_ (remove trailing white space, no tabs, etc.).  
* Check code with flake8:

.. code:: sh

    flake8 src/

.. _testcoverage:

Test coverage
-------------

Modification of code submitted through PRs should modify existing tests or create new tests where appropriate. Ideally these tests should be written to fail before the PR and pass afterwards. 
Tests should be run locally before pushing a PR, in order to save CI resources.

Some additional packages are required for running the tests, which can be installed via

.. code:: sh

    pip install scikit-digital-health[dev]

Tests should provide 100% coverage in an ideal case. Coverage can be measured by installing `coverage <https://coverage.readthedocs.io/en/coverage-5.3/>`_ and running:

.. code:: sh

    coverage run -m pytest && coverage html

from the top-level ``scikit-digital-health`` folder. This will generate html documents in ``coverage/`` which can easily be explored to find statements that are not covered:

.. code:: sh

    open coverage/index.html

.. _buildingdocs:

Building docs
-------------

To build the html documentation:

.. code:: sh

    make html 

HTML files will be generated in ``docs/_build/html/``. Since the documentation is based on docstrings, the correct version of ``scikit-digital-health`` must be installed in the current environment used to run sphinx.

Additional requirements for building the docs can be installed via (pip)

.. code:: sh

    pip install scikit-digital-health[docs]

The documentation includes mathematical formulae with LaTeX formatting. A working LaTeX document production system
(e.g. `texlive <https://www.tug.org/texlive/>`__) is required for the proper rendering of the LaTeX math in the documentation.

Fixing Warnings
~~~~~~~~~~~~~~~

-  "citation not found: R###" There is probably an underscore after a
   reference in the first line of a docstring (e.g. [1]\_). Use this
   method to find the source file: $ cd doc/build; grep -rin R####

-  "Duplicate citation R###, other instance in..."" There is probably a
   [2] without a [1] in one of the docstrings

Specific contribution guidelines
================================

.. toctree::
   :maxdepth: 1

   adding_modules
   adding_documentation
   adding_tests
