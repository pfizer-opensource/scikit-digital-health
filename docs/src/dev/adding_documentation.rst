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

Documentation of the your new module occurs almost fully in the ``__init__.py`` file
for the module. This will contain any information on the module, how it works, as
well as any relevant background information (e.g. :ref:`Gait <skdh-gait>`). For
another good reference from NumPy, see the
`numpy.fft <https://numpy.org/doc/stable/reference/routines.fft.html>`_ page.

Next is the file directing Sphinx to automatically find and document the custom
module, ``docs/ref/custom_module.rst``. Finally, to make the new module show up
in the API reference, add it to the ``docs/ref/index.rst`` toctree!

.. tab-set::

    .. tab-item:: src/skdh/custom_module/\_\_init\_\_.py
        :selected:

        .. code-block:: python

            """
            IMU <Custom Module> (:mod:`skdh.custom_module`)
            ===============================================

            .. currentmodule:: skdh.custom_module

            Inertial sensor <custom module>
            -------------------------------

            .. _Remove the `` below

            .. ``autosummary``::
                :toctree: generated/

                CustomClass  .. _this is the name of your class

            Headline 2
            ----------
            content
            """
            from skdh.custom_module.custom_module import CustomModule

    .. tab-item:: docs/ref/custom_module.rst

        .. code:: rst

            ..
                _Give it a custom explicit reference label

            .. _skdh custom-module:

            .. automodule:: skdh.custom_module
                :ignore-module-all:

    .. tab-item:: docs/ref/index.rst

        .. code:: rst

            .. _skdh api reference:

            API Reference
            =============

            .. toctree::
                :maxdepth: 2

                gait
                sit2stand
                read
                custom_module

And thats it! Before pushing and creating a pull request, make sure that the documentation builds properly:

* Make sure all the requirements are installed (``pyproject.toml docs dependencies``)
* Run the following in the terminal:

.. code:: sh

    # make sure you are in the docs folder
    cd docs
    # generate the html documentation
    make html
    # inspect the generated docs
    open _build/html/index.html
