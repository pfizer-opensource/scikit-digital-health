"""
Ambulation detection via LGBM classifier

Yiorgos Christakis
Copyright (c) 2024, Pfizer Inc. All rights reserved.
"""

from warnings import warn
from importlib import resources

from numpy import asarray, ascontiguousarray, mean, diff, concatenate, linalg
from scipy.signal import butter, sosfiltfilt
import lightgbm as lgb

from skdh.base import BaseProcess, handle_process_returns
from skdh.utility import get_windowed_view
from skdh.utility.internal import apply_resample, rle
from skdh.features import (
    Bank,
    Mean,
    MeanCrossRate,
    StdDev,
    Kurtosis,
    Skewness,
    Range,
    IQR,
    Autocorrelation,
    LinearSlope,
    SignalEntropy,
    SampleEntropy,
    PermutationEntropy,
    RangeCountPercentage,
    JerkMetric,
    SPARC,
    DominantFrequency,
    DominantFrequencyValue,
    PowerSpectralSum,
    SpectralFlatness,
    SpectralEntropy,
    DetailPower,
    DetailPowerRatio,
)


def _resolve_path(mod, file):
    return resources.files(mod) / file


class Ambulation(BaseProcess):
    """
    Processes accelerometer data to extract bouts of ambulation.

    The ambulation detection module is an algorithm for detecting ambulation or
    gait-like activities from raw tri-axial accelerometer data. The model used is
    a binary machine learning classifier that takes several signal features as input
    and produces predictions in three second non-overlapping epochs.

    The original purpose of this module was to aid in context detection for nocturnal
    scratch detection. The scratch model is specifically trained to differentiate
    restless activity from scratch activity during sleep, and therefore has issues
    dealing with unseen activities such as walking. These activities sometimes occur
    during at home recordings and increase the false positive rate of the scratch
    module. The ambulation detection module was designed to help exclude these activities
    and therefore improve the performance of the scratch module by minimizing false
    positives.

    An additional use of this model has been as a base-level context detection tool
    for step detection. Unlike the gait-bout detection algorithm, the ambulation
    detection module is not built to detect well-defined bouts of consistent gait
    for gait parameter estimation. It is rather a tool for detecting the presence
    of walking-similar activities, which makes it a good candidate for performing
    context detection for step counting.

    Input requirements:

    1. Accelerometer data must be collected with a sampling frequency of at least
    20hz. If the frequency is greater than 20hz the signal will be automatically
    downsampled.

    2. Accelerometer data is expected to be tri-axial. Orientation does not affect
    algorithm performance.

    3. Acceleration units are expected to be in G's.

    4. A minimum of 60 samples (or the equivalent of a single 3-second window) is
    required for predictions to be made.

    Parameters
    ----------
    pthresh : float
        Probability threshold for the classifier.
    """

    def __init__(self, pthresh=0.65):
        super().__init__(pthresh=pthresh)
        self.pthresh = pthresh

    @handle_process_returns(results_to_kwargs=False)
    def predict(self, time, accel, fs=None, tz_name=None, **kwargs):
        """
        predict(time, accel, fs=None)

        Function to detect ambulation or gait-like activity from 20hz accelerometer
        data collected on the wrist.

        Parameters
        ----------
        time : numpy.ndarray
            (N, ) array of unix timestamps, in seconds.
        accel : numpy.ndarray
            (N, 3) array of acceleration, in units of 'g', collected at 20hz.
        fs : float, optional
            Sampling rate. Default None. If not provided, will be inferred.

        Other Parameters
        ----------------
        tz_name : {None, str}, optional
            IANA time-zone name for the recording location if passing in `time` as
            UTC timestamps. Can be ignored if passing in naive timestamps.

        Returns
        -------
        results : dict
            Results dictionary including 3s epoch level predictions, probabilities,
            and unix timestamps.
        dict
            Results dictionary for downstream pipeline use including start and stop times for ambulation bouts.

        """
        super().predict(
            expect_days=False,
            expect_wear=False,
            time=time,
            accel=accel,
            fs=fs,
            tz_name=tz_name,
            **kwargs,
        )
        # check that input matches expectations, downsample to 20hz if necessary
        time_ds, accel_ds, fs = self._check_input(time, accel, fs)

        # load model
        model = self._load_model("ambulation_model.txt")

        # preprocess input data
        preprocessed = self._preprocess(accel_ds)

        # extract features
        features, _ = self._compute_features(preprocessed)

        # get predictions
        probabilities = model.predict(features)
        predictions = (probabilities > self.pthresh).astype(int)

        # compile results
        results = {
            "ambulation_3s_epochs_predictions": predictions,
            "ambulation_3s_epochs_probs": probabilities,
            "ambulation_3s_time": time_ds[::60],  # every 60th sample (3s at 20hz)
            "tz_name": tz_name,
        }

        # get starts and stops of ambulation bouts
        # window size of each prediction in original sampling freq
        win_len = int(fs * 3)

        # run length encoding of predictions @ 3s
        lengths_3s, starts_3s, values = rle(predictions)

        # subset to only ambulation bouts and compute stops
        starts, lengths = asarray(starts_3s[values == 1]), asarray(
            lengths_3s[values == 1]
        )
        stops = starts + lengths

        # convert to original index freq
        starts *= win_len
        stops *= win_len

        # handle cases: no bouts detected, or end is end of array
        if len(starts) and stops[-1] == time.size:
            stops[-1] = time.size - 1

        # create a single ambulation bouts array with correct units for indexing
        if len(starts):
            ambulation_bouts = concatenate((starts, stops)).reshape((2, -1)).T
        else:
            ambulation_bouts = None

        return results, {"ambulation_bouts": ambulation_bouts}

    @staticmethod
    def _preprocess(accel):
        """
        Preprocess acceleration signal:

            1. Construct a non-overlapping 3-second-windowed view of the data.
            2. Compute signal vector magnitude.
            3. High-pass filter with cutoff at 0.25hz.

        Parameters
        ----------
        accel : array-like
            Numpy array of triaxial accelerometer data. Frequency - 20hz. Units - G's.

        Returns
        -------
        preprocessed : array
            Preprocessed signal.

        """
        c_contiguous = ascontiguousarray(accel)
        windowed_accel = get_windowed_view(c_contiguous, 60, 60)

        # Vector magnitude
        x_mag = linalg.norm(windowed_accel, axis=2)

        # High-pass filter at .25hz
        sos = butter(N=1, Wn=2 * 0.25 / 20.0, btype="highpass", output="sos")
        preprocessed = sosfiltfilt(sos, x_mag, axis=1)

        return preprocessed

    @staticmethod
    def _compute_features(preprocessed):
        """
        Computes the following signal features on the preprocessed (magnitude) signal:

        - MAG_Mean()
        - MAG_MeanCrossRate()
        - MAG_StdDev()
        - MAG_Skewness()
        - MAG_Kurtosis()
        - MAG_Range()
        - MAG_IQR()
        - MAG_Autocorrelation(lag=1, normalize=True)
        - MAG_Autocorrelation(lag=5, normalize=True)
        - MAG_Autocorrelation(lag=10, normalize=True)
        - MAG_Autocorrelation(lag=20, normalize=True)
        - MAG_LinearSlope()
        - MAG_SignalEntropy()
        - MAG_SampleEntropy(m=4, r=1.0)
        - MAG_PermutationEntropy(order=3, delay=1, normalize=False)
        - MAG_RangeCountPercentage(range_min=0, range_max=1.0)
        - MAG_JerkMetric()
        - MAG_SPARC(padlevel=4, fc=10.0, amplitude_threshold=0.05)
        - MAG_DominantFrequency(padlevel=2, low_cutoff=0.0, high_cutoff=5.0)
        - MAG_DominantFrequencyValue(padlevel=2, low_cutoff=0.0, high_cutoff=5.0)
        - MAG_PowerSpectralSum(padlevel=2, low_cutoff=0.0, high_cutoff=5.0)
        - MAG_SpectralFlatness(padlevel=2, low_cutoff=0.0, high_cutoff=5.0)
        - MAG_SpectralEntropy(padlevel=2, low_cutoff=0.0, high_cutoff=5.0)
        - MAG_DetailPower(wavelet='coif4', freq_band=[1.0, 8.0])
        - MAG_DetailPowerRatio(wavelet='coif4', freq_band=[1.0, 8.0])

        Parameters
        ----------
        preprocessed : array-like
            Numpy array of preprocessed accelerometer data.

        Returns
        -------
        features, names : tuple
            Numpy array of signal features and their names.

        """
        FB = Bank()

        # add features
        FB.add(Mean())
        FB.add(MeanCrossRate())
        FB.add(StdDev())
        FB.add(Skewness())
        FB.add(Kurtosis())
        FB.add(Range())
        FB.add(IQR())
        for i in 1, 5, 10, 20:
            FB.add(Autocorrelation(lag=i, normalize=True))
        FB.add(LinearSlope())
        FB.add(SignalEntropy())
        FB.add(SampleEntropy(m=4, r=1.0))
        FB.add(PermutationEntropy(order=3, delay=1, normalize=True))
        FB.add(RangeCountPercentage(range_min=0, range_max=1.0))
        FB.add(JerkMetric())
        FB.add(SPARC())
        FB.add(DominantFrequency())
        FB.add(DominantFrequencyValue())
        FB.add(PowerSpectralSum())
        FB.add(SpectralFlatness())
        FB.add(SpectralEntropy())
        FB.add(DetailPower(wavelet="coif4", freq_band=[1.0, 8.0]))
        FB.add(DetailPowerRatio(wavelet="coif4", freq_band=[1.0, 8.0]))

        features = FB.compute(preprocessed, fs=20.0)
        names = [f"MAG_{f!r}" for f in FB._feats]

        return features.T, names

    @staticmethod
    def _check_input(time, accel, fs=None):
        """
        Checks that input meets requirements (see class docstring). Downsamples data >20hz to 20hz.
        Does not check accelerometer units.

        Parameters
        ----------
        time : array-like
            Numpy array of unix timestamps. Units of seconds.
        accel : array-like
            Numpy array of triaxial accelerometer data.
        fs : float, optional
            Sampling rate. Default None. If not provided, will be inferred.

        Returns
        -------
        time_ds : array-like
        accel_ds : array-like
        fs : float
            Detected original sampling frequency.

        """
        # check # of columns
        _, c = accel.shape
        if not (c == 3):
            raise ValueError(f"Input expected to have 3 columns, but found {str(c)}")

        # check fs & downsample if necessary
        fs = round(1 / mean(diff(time)), 3) if fs is None else fs
        if fs < 20.0:
            raise ValueError(f"Frequency ({fs:.2f}Hz) is too low (<20Hz).")
        elif fs > 20.0:
            warn(
                "Frequency is > 20Hz. Downsampling to 20Hz.",
                UserWarning,
            )
            time_ds, (accel_ds,) = apply_resample(
                time=time,
                goal_fs=20.0,
                data=(accel,),
                fs=fs,
            )
        else:
            time_ds = time
            accel_ds = accel

        # check input length
        r, _ = accel_ds.shape
        if not (r >= 60):
            raise ValueError(
                f"Input at 20hz expected to have at least 60 rows, but found {str(r)}"
            )

        return time_ds, accel_ds, fs

    @staticmethod
    def _load_model(model_path):
        """
        Loads the machine learning model.

        Parameters
        ----------
        model_path : str

        Returns
        -------
        model : object

        """
        # load the classification model
        lgb_file = str(_resolve_path("skdh.context.model", model_path))
        model = lgb.Booster(model_file=lgb_file)
        return model
