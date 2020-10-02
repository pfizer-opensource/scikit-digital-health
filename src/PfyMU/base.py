"""
Base classes, functions, etc for the PfyMU library

Lukas Adamowicz
Pfizer DMTI 2020
"""


class _BaseProcess:
    # names of the variables that are passed to predict
    # CHANGE IF predict/_predict call changes!
    _file = 'file'
    _time = 'time'
    _acc = 'accel'
    _gyro = 'gyro'
    _mag = 'magnet'
    _temp = 'temperature'
    _days = 'day_ends'

    def __str__(self):
        return self._proc_name.replace(' ', '')

    def __init__(self, name, return_result=True):
        """
        Intended to be subclassed

        Parameters
        ----------
        name : str
            Name of the process
        return_result : bool, optional
            Return the result part of predict, or the input/output dictionary
        """
        self._proc_name = name
        self._return_result = return_result

    def predict(self, *args, **kwargs):
        result = self._predict(*args, **kwargs)

        if self._return_result:
            return result[1]
        else:
            return result[0]

    def _predict(self, *args, **kwargs):
        return None, None
