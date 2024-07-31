from numpy import ndarray, nonzero, diff

from skdh.base import BaseProcess, handle_process_returns


class Completeness(BaseProcess):
    """
    Generate a summary of data completeness for data provided.

    Parameters
    ----------
    streams_to_check : list, optional
        List of data-streams to assess completeness for. If None, defaults
        to just checking the accelerometer data.
    wear_name : str, optional
        Name of the wear data stream. Defaults to "wear".
    """
    def __init__(self, streams_to_check=None, wear_name="wear"):
        super().__init__(
            streams_to_check=streams_to_check,
            wear_name=wear_name,
        )

        self.streams_to_check = streams_to_check if streams_to_check is not None else [self._acc]
        self.wear_name = wear_name
    
    def handle_data(self, time, *, day_ends=None, wear=None, **data):
        """
        Handle the data streams

        Parameters
        ----------
        data : _type_
            _description_
        day_ends : _type_
            _description_
        wear : _type_
            _description_
        """
        # separate out any data that corresponds to this time array
        this_data = {k: data.pop(k) for k in data if isinstance(data[k], ndarray) and data[k].shape == time.shape}

        # figure out the gaps
        # TODO have this check for filled gaps as well
        gaps = 

    @handle_process_returns(results_to_kwargs=False)
    def predict(self, time, **kwargs):
        """
        Generate completeness results

        Parameters
        ----------
        time : numpy.ndarray
            Time data for any array in `**data` that is not a dictionary and does
            not have its own time series.
        **data : numpy.ndarray, dict
            Data to be checked for completeness. Either a numpy array which should 
            align with `time` for size/number of samples, or a dictionary of data
            containing a time series array and its associated timestamps.
        """
        super().predict(expect_days=False, expect_wear=False, **kwargs)

        # get an array of data that we need to check for completeness
        data = {k: kwargs[k] for k in self.streams_to_check}

        # get a few arrays out of kwargs that we know dont need completeness
        day_ends = kwargs.get("day_ends", None)
        wear = kwargs.get(self.wear_name, None)

        self.handle_data(time, day_ends, wear, **data)

