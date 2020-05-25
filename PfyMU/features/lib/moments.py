"""
Features from statistical moments

Lukas Adamowicz
Pfizer DMTI 2020
"""
from numpy import mean

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
            Columns to use if providing a """
        super()._compute(x, *args)

        self._result = mean(x, axis=1)


