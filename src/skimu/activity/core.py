"""
Core functionality for activity level segregation

Lukas Adamowicz
Pfizer DMTI 2020
"""
from skimu.activity.core import roll_mean, roll_median_1, angle, hfen_plus, enmo, enmoa


class Metrics:
    def __init__(self, short_win_length=5.0, metrics='hfen+'):
        """
        Compute various metrics on acceleration data.

        Parameters
        ----------
        short_win_length : float, optional
            Window length in seconds for the short window. Default is 5.0s
        metrics : {'hfen+', 'enmo', 'enmoa', 'anglex', 'angley', 'anglez'}, array_like, optional
            Metrics to compute in the data. Default is 'hfen+'. Can be an array-like of metrics as
            well. See notes for detailed description of metrics

        Notes
        -----
        The metrics are as follows:
        - hfen+: High-pass Filter followed by Euclidean Norm, plus the euclidean norm of the
        low-pass filtered acceleration minus 1g
        - enmo: Euclidean Norm of the acceleration, minus 1g
        - enmoa: Euclidean Norm of the acceleration, minus 1g. Additionally, negative values are
        clipped to 0.
        - angle<_>: Angle in the <_> axis

        Examples
        --------
        >>> metrics = Metrics(short_win_length=5.0, metrics=['hfen+', 'enmoa', 'anglex'])
        >>> hfen_p, enmoa, anglex = metrics.compute(acceleration, fs)
        """
        self.win_l = short_win_length
        if isinstance(metrics, str):
            self.metrics = [metrics]
        else:
            self.metrics = metrics

    def _compute(self, acceleration, fs):
        """
        Returns
        -------
        metric_1, metric_2, ... : numpy.ndarray
            Arrays of the compute metrics, taking the mean value over non-overlapping windows of
            length `short_win_length`. Returned in the order of `metrics`, as seperate arrays
        """
        ret = {}
        for metr in self.metrics:
            if metr.lower() == 'hfen+':
                ret[metr] = roll_mean(hfen_plus(acceleration, fs, cut=0.2, N=4), fs, self.win_l)
            elif metr.lower() == 'enmo':
                ret[metr] = roll_mean(enmo(acceleration), fs, self.win_l)
            elif metr.lower() == 'enmoa':
                ret[metr] = roll_mean(enmoa(acceleration), fs, self.win_l)
            elif 'angle' in metr.lower():
                ax = metr.lower()[-1]
                # take the mean of windows, after computing the angle of the rolling median with
                # window_step = 1
                ret[metr] = roll_mean(angle(roll_median_1(acceleration, fs), ax), fs, self.win_l)

        return (ret[i] for i in self.metrics)
