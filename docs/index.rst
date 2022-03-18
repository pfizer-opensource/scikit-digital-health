..
   _skdh documentation master file, created by
   sphinx-quickstart on Tue Oct  6 14:50:59 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Scikit Digital Health
=====================

.. image:: https://github.com/PfizerRD/scikit-digital-health/workflows/skdh/badge.svg
   :alt: GitHub Actions Badge

`SciKit-Digital-Health` is a Python package with methods for reading, pre-processing, manipulating, and analyzing Inertial Meausurement Unit data. `scikit-digital-health` contains the following sub-modules:

.. panels::
    :img-top-cls: pl-3 pr-3
    :card: shadow

    :img-top: _static/skdh_io.svg

    :badge:`skdh.io,badge-primary`

    +++
    .. link-button:: skdh io
        :type: ref
        :text: Ingestion of common binary data files and generic data stores.
        :classes: btn-outline-primary stretched-link btn-block

    ---
    :img-top: _static/skdh_preproc.svg
    :badge:`skdh.preprocessing,badge-primary`
    +++
    .. link-button:: skdh preprocessing
        :type: ref
        :text: Pre-processing algorithms for inertial data.
        :classes: btn-outline-primary stretched-link btn-block

    ---
    :img-top: _static/skdh_gait.svg
    :badge:`skdh.gait,badge-primary`
    +++
    .. link-button:: skdh gait
        :type: ref
        :text: Gait detection and analysis from lumbar inertial data.
        :classes: btn-outline-primary stretched-link btn-block

    ---
    :img-top: _static/skdh_s2s.svg
    :badge:`skdh.sit2stand,badge-primary`
    +++
    .. link-button:: skdh sit2stand
        :type: ref
        :text: Sit-to-stand detection and analysis from lumbar inertial data.
        :classes: btn-outline-primary stretched-link btn-block

    ---
    :img-top: _static/skdh_activity.svg
    :badge:`skdh.activity,badge-primary`
    +++
    .. link-button:: skdh activity
        :type: ref
        :text: Actigraphy & activity analysis from inertial data.
        :classes: btn-outline-primary stretched-link btn-block

    ---
    :img-top: _static/skdh_sleep.svg
    :badge:`skdh.sleep,badge-primary`
    +++
    .. link-button:: skdh sleep
        :type: ref
        :text: Sleep detection and analysis from wrist inertial data.
        :classes: btn-outline-primary stretched-link btn-block

    ---
    :img-top: _static/skdh_features.svg
    :badge:`skdh.features,badge-primary`
    +++
    .. link-button:: skdh features
        :type: ref
        :text: Common time-series signal features.
        :classes: btn-outline-primary stretched-link btn-block

    ---
    :img-top: _static/skdh_utility.svg
    :badge:`skdh.utility,badge-primary`
    +++
    .. link-button:: skdh utility
        :type: ref
        :text: Utility functions for time-series analysis.
        :classes: btn-outline-primary stretched-link btn-block


..
   _ keep the toctree hidden for a cleaner landing page

.. toctree::
   :maxdepth: 2
   :hidden:

   src/installation
   src/usage
   src/dev/contributing
   ref/index



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
