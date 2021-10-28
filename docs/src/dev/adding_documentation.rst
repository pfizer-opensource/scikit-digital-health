.. _adding-documentation:

####################
Adding documentation
####################

Documentation should be fairly simple as long as the class/function docstrings are correctly written.

Style guidelines
----------------

* ``scikit-digital-health`` uses `NumPy Docstrings <https://numpydoc.readthedocs.io/en/latest/format.html>`_

Adding documentation
--------------------

The one place to be careful when documenting classes is to make sure that the class docstring occurs directly below the class definition. 
Other than this, the docstrings will be automatically scanned and generated, with the appropriate call in ``__init__.py``.

Documentation of the your new module occurs almost fully in the ``__init__.py`` file for the module. An example for the example ``preprocessing`` module is below:

.. code-block:: python

    # src/skdh/preprocessing/__init__.py
    """
    IMU Preprocessing (:mod:`skdh.preprocessing`)
    ==============================================

    .. currentmodule:: skdh.preprocessing

    Inertial sensor preprocessing
    -----------------------------

    .. autosummary::
        :toctree: generated/

        Preprocessing  .. _this is the name of your class
    
    Headline 2
    ----------
    content
    """
    from skdh.preprocessing.preprocessing import Preprocessing

The docstring is written like a `.rst` header file (which is how it will get interpreted). For an example in `skdh`, see the `gait init file <src/skdh/gait/__init__.py>`_.  For an example of a good module documentation from NumPY, see the `FFT <https://numpy.org/doc/stable/reference/routines.fft.html>`_ page.

With this documentation written, the last thing is to add a short `.rst` file inside the actual documentation folder, which will instruct `sphinx` to read the documentation for this module:

.. code:: rst

    .. _this_file: docs/ref/preprocessing.rst

    .. automodule:: skdh.preprocessing
        :ignore-module-all:

Then add this new file to ``docs/ref/index.rst``:

.. code:: rst

    .. _this_file: docs/ref/index.rst
    .. _skdh api reference

    API Reference
    =============

    .. toctree::
        :maxdepth: 2

        gait
        sit2stand
        read
        preprocessing

And thats it! Before pushing and creating a pull request, make sure that the documentation builds properly:

* Make sure all the requirements are installed (``doc_requirements.txt``)
* Run the following in the terminal:

.. code:: sh

    # make sure you are in the docs folder
    cd docs
    # generate the html documentation
    make html
    # inspect the generated docs
    open _build/html/index.html
