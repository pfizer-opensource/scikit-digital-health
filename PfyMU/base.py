"""
Base classes, functions, etc for the PfyMU library

Lukas Adamowicz
Pfizer DMTI 2020
"""


class _BaseProcess:
    def __init__(self, name):
        """
        Intended to be subclassed

        Parameters
        ----------
        name : str
            Name of the process
        """
        self.name = name

    def predict(self, file=None, time=None, accel=None, gyro=None, temperature=None, **kwargs):
        result = self._predict(file=file, time=time, accel=accel, gyro=gyro, temperature=temperature, **kwargs)
        return result[1]

    def _predict(self, *, file=None, time=None, accel=None, gyro=None, temperature=None, **kwargs):
        pass
