"""
Core functionality for activity level segregation

Lukas Adamowicz
Pfizer DMTI 2020
"""


class Metrics:
    def __init__(self, metrics='hfen+'):
        """
        Compute various metrics on acceleration data.

        Parameters
        ----------
        metrics : {'hfen+', 'enmo', 'enmoz', 'anglex', 'angley', 'anglez'}, array_like, optional
            Metrics to compute in the data. Default is 'hfen+'. Can be an array-like of metrics as well. See notes
            for detailed description of metrics

        Notes
        -----
        The metrics are as follows:
        - hfen+: High-pass Filter followed by Euclidean Norm, plus the euclidean norm of the low-pass filtered
        acceleration minus 1g
        - enmo: Euclidean Norm of the acceleration, minus 1g
        - enmoz: Euclidean Norm of the acceleration, minus 1g. Additionally, negative values are clipped to 0
        - angle<_>: Angle in the <_> axis
        """
        if isinstance(metrics, str):
            self.metrics = [metrics]
        else:
            self.metrics = metrics

