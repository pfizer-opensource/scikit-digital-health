from numpy import ndarray, median, diff

from skdh.base import BaseProcess, handle_process_returns
from skdh.utility.internal import fill_data_gaps


class FillGaps(BaseProcess):
    """
    Fill gaps in data so that the data is continuous, which is what is expected
    by the rest of SKDH.

    Parameters
    ----------
    fill_values : {None, dict}, optional
        Dictionary with keys and values to fill data streams with. This will determine
        which data-streams to look for to fill, outside of the main four listed
        in the Notes section. Additionally, see Notes for default values if not provided.

    Notes
    -----
    Default fill values are:

    - accel: numpy.array([0.0, 0.0, 1.0])
    - gyro: 0.0
    - temperature: 0.0
    - ecg: 0.0

    The default fill values are not NaN in order to not cause issues with filters
    or other signal processing methods where NaN values may be propagated beyond
    a data gap and effect results where data is actually available.
    """

    def __init__(self, fill_values=None):
        super().__init__(fill_values=fill_values)

        self.fill_values = {} if fill_values is None else fill_values

    @handle_process_returns(results_to_kwargs=True)
    def predict(self, *, time, **kwargs):
        """
        predict(*, time, accel=None, fs=None, gyro=None, temperature=None, ecg=None, **kwargs)
        Fill gaps in data streams.

        Parameters
        ----------
        time : numpy.ndarray
            (N,) array of timestamps (unix seconds).
        accel : numpy.ndarray, optional
            (N, 3) array of acceleration data.
        fs : float, optional
            Sampling frequency in Hz for the acceleration data.
        gyro : numpy.ndarray, dict, optional
            (N, 3) array of gyroscope data, or a dictionary containing the keys
            'time', and 'values' if not using the same timestamps (`time`) as
            `accel`.
        temperature : numpy.ndarray, dict, optional
            (N,) array of temperature data, or a dictionary containing the keys
            'time', and 'values' if not using the same timestamps (`time`) as
            `accel`.
        ecg : numpy.ndarray, dict, optional
            (N,) array of ECG data, or a dictionary containing the keys
            'time', and 'values' if not using the same timestamps (`time`) as
            `accel`.
        **kwargs : numpy.ndarray, dict, optional
            Any additional data streams that need to be filled. These will only be
            filled if they are given fill values with `fill_values`.
        """
        self.fill_values.setdefault("accel", [0.0, 0.0, 1.0])
        self.fill_values.setdefault("gyro", 0.0)
        self.fill_values.setdefault("temperature", 0.0)
        self.fill_values.setdefault("ecg", 0.0)

        super().predict(expect_days=False, expect_wear=False, time=time, **kwargs)

        fs = kwargs.get("fs", 1 / median(diff(time[:5000])))

        # get a list of all the streams we can have
        streams = self.fill_values.keys()

        # get the streams that are aligned with the main data (ie time), and
        # also single streams that are not aligned with the main data
        aligned_streams = [i for i in streams if isinstance(kwargs.get(i), ndarray)]
        single_streams = [i for i in streams if isinstance(kwargs.get(i), dict)]

        self.logger.info(f"Aligned streams to fill gaps in: {aligned_streams}")
        self.logger.info(f"Single streams to fill gaps in: {single_streams}")

        # fill gaps in the aligned data
        time_rs, data_rs = fill_data_gaps(
            time, fs, self.fill_values, **{i: kwargs[i] for i in aligned_streams}
        )

        # fill gaps in the single data streams
        for stream in single_streams:
            s_time_rs, s_data_rs = fill_data_gaps(
                kwargs[stream]["time"],
                kwargs[stream].get(
                    "fs", 1 / median(diff(kwargs[stream]["time"][:5000]))
                ),
                self.fill_values,
                **{stream: kwargs[stream]["values"]},
            )
            s_data_rs["time"] = s_time_rs
            # rename the stream key to values
            s_data_rs["values"] = s_data_rs.pop(stream)
            # put the re-sampled single stream data into the results dictionary
            data_rs[stream] = s_data_rs

        # add time back into the re-sampled results
        data_rs[self._time] = time_rs

        return data_rs
