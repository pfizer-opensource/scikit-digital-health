"""
Features from statistical moments

Lukas Adamowicz
Pfizer DMTI 2020
"""
from numpy import mean, std

from PfyMU.features.core import Feature


class Mean(Feature):
    def __init__(self):
        """
        Compute the signal mean.

        Examples
        --------
        >>> signal = np.arange(15).reshape((5, 3))
        >>> mn = features.Mean()
        >>> mn.compute(signal)
        array([6., 7., 8.])
        """
        super().__init__('Mean', {})

    def _compute(self, x, *args):
        doc = """
        Compute the signal mean.
        
        Parameters
        ----------
        signal : {numpy.ndarray, pandas.DataFrame}
            ndarray (up to 3D) or DataFrame with the signal
        columns : array-like, optional
            Columns to use if providing a pandas dataframe
            
        Returns
        -------
        mean : {numpy.ndarray, pandas.DataFrame}
            ndarray or DataFrame with the mean of the signal.
        """
        self.compute.__doc__ = doc  # overwrite the compute method doc
        super()._compute(x, *args)

        self._result = mean(x, axis=1)


class StdDev(Feature):
    def __init__(self):
        """
        Compute the signal standard deviation.

        Examples
        --------
        >>> signal = np.arange(15).reshape((5, 3))
        >>> features.StDev().compute(signal)
        array([[4.74341649, 4.74341649, 4.74341649]])
        """
        super().__init__('StdDev', {})

    def _compute(self, x, *args):
        super()._compute(x, *args)

        self._result = std(x, axis=1, ddof=1)
