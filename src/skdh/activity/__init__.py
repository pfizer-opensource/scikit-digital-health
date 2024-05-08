r"""
IMU Activity Analysis (:mod:`skdh.activity`)
============================================

.. currentmodule:: skdh.activity

Pipeline Activity Processing
----------------------------

.. autosummary::
    :toctree: generated/

    ActivityLevelClassification

.. _accelerometer-metrics:

Activity Endpoints
------------------

.. autosummary::
    :toctree: generated/

    ActivityEndpoint
    IntensityGradient
    MaxAcceleration
    TotalIntensityTime
    BoutIntensityTime
    FragmentationEndpoints
    EqualAverageDurationThreshold
    SignalFeatures

Accelerometer Metrics
---------------------

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

Activity level classification is the process of using accelerometer metrics to derive
estimates of the time spend in different energy expenditure states, typically classified
by Metabolic Equivalent of Task (MET). The MET is the rate of energy expenditure for an
individual of a certain mass performing physical activities, relative to a baseline.
The different classifications are typically

- sedentary
- light (< 3 MET)
- moderate (3-6 MET)
- vigorous (> 6 MET)

Different research has yielded different methods of estimating these thresholds, with
different acceleration metrics and cutpoints for those metrics. The ones available as
part of Scikit-Digital-Health are the following:

- ``"esliger_lwrist_adult"`` [1]_ Note that these use light and moderate thresholds of
    4 and 7 METs
- ``"esliger_rwirst_adult"`` [1]_ Note that these use light and moderate thresholds of
    4 and 7 METs
- ``"esliger_lumbar_adult"`` [1]_ Note that these use light and moderate thresholds of
    4 and 7 METs
- ``"schaefer_ndomwrist_child6-11"`` [2]_
- ``"phillips_rwrist_child8-14"`` [3]_
- ``"phillips_lwrist_child8-14"`` [3]_
- ``"phillips_hip_child8-14"`` [3]_
- ``"vaha-ypya_hip_adult"`` [4]_ Note that this uses the MAD metric, and originally
    used 6 second long windows
- ``"hildebrand_hip_adult_actigraph"`` [5]_, [6]_
- ``"hildebrand_hip_adult_geneactv"`` [5]_, [6]_
- ``"hildebrand_wrist_adult_actigraph"`` [5]_, [6]_
- ``"hildebrand_wrist_adult_geneactiv"`` [5]_, [6]_
- ``"hildebrand_hip_child7-11_actigraph"`` [5]_, [6]_
- ``"hildebrand_hip_child7-11_geneactiv"`` [5]_, [6]_
- ``"hildebrand_wrist_child7-11_actigraph"`` [5]_, [6]_
- ``"hildebrand_wrist_child7-11_geneactiv"`` [5]_, [6]_
- ``"migueles_wrist_adult"`` [7]_ **these are the default cutpoints used**

The thresholds have been automatically scaled to the average values, and can be used
with any length windows (though most originally use 1s windows), and use the appropriate
acceleration metric.

.. _Using Custom Cutpoints:

Adding Custom Endpoints
-----------------------
Custom endpoints are simple to add - each custom endpoint should be a subclass
of the :class:`.ActivityEndpoint`. Endpoint classes can generate multiple endpoints
at once, but the names for all of the endpoints need to be specified in the custom
class, as this is how the results dictionary is populated.

.. code-block:: python

    from skdh.activity import ActivityEndpoint

    class CustomEndpointSingle(ActivityEndpoint):
        def __init__(self, arg1, arg2=None, state='wake'):
            super().__init__("custom endpoint name", state)

            self.arg1 = arg1
            self.arg2 = arg2

        def predict(self, results, i, accel_metric, epoch_s, epochs_per_min, **kwargs):
            super().predict()

            # desired processing

            # save the results
            results[self.name][i] = custom_endpoint_res

    class CustomEndpointMultiple(ActivityEndpoint):
        def __init__(self, arg1, state='wake'):
            super().__init__(
                [
                    "custom ept 1",
                    "custom ept 2",
                ],
                state
            )

The only required parameter for the custom endpoint class `__init__` is `state`,
which should be set when initialized to the state in which the endpoint should
be calculated. The `.predict` method is run for every block of wear data during
`state`, meaning it could potentially get run multiple times during the same day.
For some custom endpoints, this may not be a problem, however if one value
needs to be calculated on all the data for the day, the state of the class
is kept/left alone between runs. This leads to the below example, which saves
several values between runs until the full day is done (at which point `.reset_cache()`
is called:

.. code-block:: python

    class IntensityGradient(ActivityEndpoint):
        def __init__(self, state="wake"):
            super(IntensityGradient, self).__init__(
                ["intensity gradient", "ig intercept", "ig r-squared"], state
            )
            # default from rowlands
            self.ig_levels = (
                array([i for i in range(0, 4001, 25)] + [8000], dtype="float") / 1000
            )
            self.ig_vals = (self.ig_levels[1:] + self.ig_levels[:-1]) / 2

            # values that need to be cached and stored between runs
            self.hist = zeros(self.ig_vals.size)
            self.ig = None
            self.ig_int = None
            self.ig_r = None
            self.i = None

        def predict(self, results, i, accel_metric, epoch_s, epochs_per_min, **kwargs):
            super(IntensityGradient, self).predict()
            # get the counts in number of minutes in each intensity bin
            self.hist += (
                histogram(accel_metric, bins=self.ig_levels, density=False)[0]
                / epochs_per_min
            )
            # get pointers to the intensity gradient results
            self.ig = results[self.name[0]]
            self.ig_int = results[self.name[1]]
            self.ig_r = results[self.name[2]]
            self.i = i

        def reset_cached(self):
            super(IntensityGradient, self).reset_cached()
            # make sure we have results locations to set
            if all([i is not None for i in [self.ig, self.ig_int, self.ig_r, self.i]]):
                # compute the results
                # convert back to mg to match existing work
                lx = log(self.ig_vals[self.hist > 0] * 1000)
                ly = log(self.hist[self.hist > 0])
                if ly.size <= 1:
                    slope = intercept = rval = nan
                else:
                    slope, intercept, rval, *_ = linregress(lx, ly)
                # set the results values
                self.ig[self.i] = slope
                self.ig_int[self.i] = intercept
                self.ig_r[self.i] = rval ** 2
            # reset the histogram counts to 0, and results to None
            self.hist = zeros(self.ig_vals.size)
            self.ig = None
            self.ig_int = None
            self.ig_r = None
            self.i = None

Using Custom Cutpoints/Metrics
------------------------------
If you would like to use your own custom cutpoints/metric, they can be supplied in a
dictionary as follows:

.. code-block:: python

    from skdh.activity import ActivityLevelClassification
    from skdh.utility import rolling_mean

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

    mvpa = ActivityLevelClassification(cutpoints=custom_cutpoints)

References
----------
.. [1] D. W. Esliger, A. V. Rowlands, T. L. Hurst, M. Catt, P. Murray, and R. G. Eston,
    “Validation of the GENEA Accelerometer,” Medicine & Science in Sports & Exercise,
    vol. 43, no. 6, pp. 1085–1093, Jun. 2011, doi: 10.1249/MSS.0b013e31820513be.
.. [2] C. A. Schaefer, C. R. Nigg, J. O. Hill, L. A. Brink, and R. C. Browning,
    “Establishing and Evaluating Wrist Cutpoints for the GENEActiv Accelerometer in
    Youth,” Med Sci Sports Exerc, vol. 46, no. 4, pp. 826–833, Apr. 2014,
    doi: 10.1249/MSS.0000000000000150.
.. [3] L. R. S. Phillips, G. Parfitt, and A. V. Rowlands, “Calibration of the
    GENEA accelerometer for assessment of physical activity intensity in children,”
    Journal of Science and Medicine in Sport, vol. 16, no. 2, pp. 124–128, Mar. 2013,
    doi: 10.1016/j.jsams.2012.05.013.
.. [4] H. Vähä-Ypyä et al., “Validation of Cut-Points for Evaluating the Intensity
    of Physical Activity with Accelerometry-Based Mean Amplitude Deviation (MAD),”
    PLOS ONE, vol. 10, no. 8, p. e0134813, Aug. 2015, doi: 10.1371/journal.pone.0134813.
.. [5] M. Hildebrand, V. T. Van Hees, B. H. Hansen, and U. Ekelund, “Age Group
    Comparability of Raw Accelerometer Output from Wrist- and Hip-Worn Monitors,”
    Medicine & Science in Sports & Exercise, vol. 46, no. 9, pp. 1816–1824, Sep. 2014,
    doi: 10.1249/MSS.0000000000000289.
.. [6] M. Hildebrand, B. H. Hansen, V. T. van Hees, and U. Ekelund, “Evaluation of raw
    acceleration sedentary thresholds in children and adults,” Scandinavian Journal of
    Medicine & Science in Sports, vol. 27, no. 12, pp. 1814–1823, 2017,
    doi: https://doi.org/10.1111/sms.12795.
.. [7] J. H. Migueles et al., “Comparability of accelerometer signal aggregation
    metrics across placements and dominant wrist cut points for the assessment of
    physical activity in adults,” Scientific Reports, vol. 9, no. 1, Art. no. 1,
    Dec. 2019, doi: 10.1038/s41598-019-54267-y.

"""

from skdh.activity.core import ActivityLevelClassification
from skdh.activity.metrics import *
from skdh.activity import metrics
from skdh.activity.cutpoints import get_available_cutpoints
from skdh.activity.endpoints import *
from skdh.activity import endpoints

__all__ = (
    ["ActivityLevelClassification", "metrics", "get_available_cutpoints", "endpoints"]
    + metrics.__all__
    + endpoints.__all__
)
