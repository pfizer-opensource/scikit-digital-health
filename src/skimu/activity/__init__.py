r"""
IMU Activity Analysis (:mod:`skimu.activity`)
=====================================

.. currentmodule:: skimu.activity

Pipeline activity processing
------------------------

.. autosummary::
    :toctree: generated/

    MVPActivityClassification

.. _accelerometer-metrics:

Accelerometer Metrics
------------------------

.. autosummary::
    :toctree: generated/

    metric_en
    metric_enmo
    metric_bfen
    metric_hfen
    metric_hfenplus
    metric_mad

Background Information
----------------------

Activity level classification is the process of using accelerometer metrics to derive estimates
of the time spend in different energy expenditure states, typically classified by Metabolic
Equivalent of Task (MET). The MET is the rate of energy expenditure for an individual of a
certain mass performing physical activities, relative to a baseline. The different classifications
are typically

- sedentary
- light (< 3 MET)
- moderate (3-6 MET)
- vigorous (> 6 MET)

Different research has yielded different methods of estimating these thresholds, with different
acceleration metrics and cutpoints for those metrics. The ones available by default are the
following:

- `"esliger_lwrist_adult"` [1]_ Note that these use light and moderate thresholds of 4 and 7 METs
- `"esliger_rwirst_adult"` [1]_ Note that these use light and moderate thresholds of 4 and 7 METs
- `"esliger_lumbar_adult"` [1]_ Note that these use light and moderate thresholds of 4 and 7 METs
- `"schaefer_ndomwrist_child6-11"` [2]_
- `"phillips_rwrist_child8-14"` [3]_
- `"phillips_lwrist_child8-14"` [3]_
- `"phillips_hip_child8-14"` [3]_
- `"vaha-ypya_hip_adult"` [4]_ Note that this uses the MAD metric, and originally used 6 second
  long windows
- `"hildebrand_hip_adult_actigraph"` [5]_, [6]_
- `"hildebrand_hip_adult_geneactv"` [5]_, [6]_
- `"hildebrand_wrist_adult_actigraph"` [5]_, [6]_
- `"hildebrand_wrist_adult_geneactiv"` [5]_, [6]_
- `"hildebrand_hip_child7-11_actigraph"` [5]_, [6]_
- `"hildebrand_hip_child7-11_geneactiv"` [5]_, [6]_
- `"hildebrand_wrist_child7-11_actigraph"` [5]_, [6]_
- `"hildebrand_wrist_child7-11_geneactiv"` [5]_, [6]_

The thresholds have been automatically scaled to the average values, and can be used with any
length windows (though most originally use 1s windows), and use the appropriate acceleration metric.

Using Custom Cutpoints/Metrics
------------------------------
If you would like to use your own custom cutpoints/metric, they can be supplied in a dictionary
as follows:

.. code-block:: python

    from skimu.activity import MVPActivityClassification
    from skimu.utility import rolling_mean

    def metric_fn(accel, wlen, \*args, \*\*kwargs):
        # compute acceleration metric for non-overlapping windows of length wlen
        metric = compute_metric()
        return rolling_mean(metric, wlen, wlen)

    custom_cutpoints = {
        "metric": metric_fn,  # function handle
        "kwargs": {"metric_fn_kwarg1": value1},
        "sedentary": sedentary_max,  # maximum value for sedentary (min value for light)
        "light": light_max,  # maximum value for light acvitity (min value for moderate)
        "moderate": moderate_max  # max value for moderate (min value for vigorous)
    }

    mvpa = MVPActivityClassification(cutpoints=custom_cutpoints)

References
----------
.. [1] D. W. Esliger, A. V. Rowlands, T. L. Hurst, M. Catt, P. Murray, and R. G. Eston,
    “Validation of the GENEA Accelerometer,” Medicine & Science in Sports & Exercise, vol. 43,
    no. 6, pp. 1085–1093, Jun. 2011, doi: 10.1249/MSS.0b013e31820513be.
.. [2] C. A. Schaefer, C. R. Nigg, J. O. Hill, L. A. Brink, and R. C. Browning, “Establishing and
    Evaluating Wrist Cutpoints for the GENEActiv Accelerometer in Youth,” Med Sci Sports Exerc,
    vol. 46, no. 4, pp. 826–833, Apr. 2014, doi: 10.1249/MSS.0000000000000150.
.. [3] L. R. S. Phillips, G. Parfitt, and A. V. Rowlands, “Calibration of the GENEA accelerometer
    for assessment of physical activity intensity in children,” Journal of Science and Medicine in
    Sport, vol. 16, no. 2, pp. 124–128, Mar. 2013, doi: 10.1016/j.jsams.2012.05.013.
.. [4] H. Vähä-Ypyä et al., “Validation of Cut-Points for Evaluating the Intensity of Physical
    Activity with Accelerometry-Based Mean Amplitude Deviation (MAD),” PLOS ONE, vol. 10, no. 8,
    p. e0134813, Aug. 2015, doi: 10.1371/journal.pone.0134813.
.. [5] M. Hildebrand, V. T. Van Hees, B. H. Hansen, and U. Ekelund, “Age Group Comparability of
    Raw Accelerometer Output from Wrist- and Hip-Worn Monitors,” Medicine & Science in Sports &
    Exercise, vol. 46, no. 9, pp. 1816–1824, Sep. 2014, doi: 10.1249/MSS.0000000000000289.
.. [6] M. Hildebrand, B. H. Hansen, V. T. van Hees, and U. Ekelund, “Evaluation of raw
    acceleration sedentary thresholds in children and adults,” Scandinavian Journal of Medicine &
    Science in Sports, vol. 27, no. 12, pp. 1814–1823, 2017, doi: https://doi.org/10.1111/sms.12795.
"""
from skimu.activity.core import MVPActivityClassification
from skimu.activity.metrics import *
from skimu.activity import metrics

__all__ = ["MVPActivityClassification", "metrics"] + metrics.__all__
