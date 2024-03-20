from skdh.base import BaseProcess, handle_process_returns


class AssessCompleteness(BaseProcess):
    """
    Module to assess completeness of data streams

    Parameters
    ----------
    """
    def __init__(
        self,
        resample_width_mins=5,
        gap_size_mins=30,
        ranges=None,
        data_gaps=None,
        time_periods=None,
        timescales=None
    ):
        super().__init__(
            resample_width_mins=resample_width_mins,
            gap_size_mins=gap_size_mins,
            ranges=ranges,
            data_gaps=data_gaps,
            time_periods=time_periods,
            timescales=timescales,
        )

        self.resample_width_mins = resample_width_mins
        self.gap_size_mins = gap_size_mins
        # TODO needs a default
        self.ranges = ranges if ranges is not None else []
        # TODO needs a default
        self.data_gaps = data_gaps if data_gaps is not None else []
        # TODO needs a default
        self.time_periods = time_periods if time_periods is not None else []
        # TODO needs a default
        self.timescales = timescales if timescales is not None else []
    
    @handle_process_returns(results_to_kwargs=False)
    def predict(self, time=None, **kwargs):
        """
        predict(time, *, accel=None, temperature=None, gyro=None, ecg=None, time_opt=None)
        Assess completeness

        Parameters
        ----------
        time : numpy.ndarray
            Array of timestamps corresponding to data.
        accel : numpy.ndarray, optional
            Array of acceleration values.
        temperature : numpy.ndarray, optional
            Array of temperature values.
        gyro : numpy.ndarray, optional
            Array of gyroscope (angular velocity) values.
        ecg : numpy.ndarray, optional
            Array of ECG reading values. 
        time_opt : dict, optional
            Optional dictionary of additional timestamp arrays for data-streams.
            Keys are the same as `predict` parameters. If a datastream does not
            have a key in `time_opt`, it will assume to be using the `time` parameter
            as its timestamp array.
        """
        super().predict(expect_days=False, expect_wear=False, time=time, **kwargs)

        # standardize the input
        streams = [self._acc, self._temp, self._gyro, 'ecg']

        data = {
            i: {
                'stream': kwargs.get(i, None),
                'time': kwargs.get('time_opt', {}).get(i, time)
            } for i in streams if i in kwargs
        }
        
