"""
Read from a numpy compressed file.

Lukas Adamowicz
Pfizer DMTI 2020
"""
from warnings import warn
from pathlib import Path

from numpy import load

from skdh.base import BaseProcess
from skdh.read.utility import FileSizeError


class ReadNumpyFile(BaseProcess):
    """
    Read a Numpy compressed file into memory. The file should have been
    created by `numpy.savez`. The data contained is read in
    unprocessed - ie acceleration is already assumed to be in units of
    'g' and time in units of seconds. No day windowing is performed. Expected
    keys are `time` and `accel`. If `fs` is present, it is used as well.
    """
    def __init__(self):
        super(ReadNumpyFile, self).__init__()

    def predict(self, file=None, **kwargs):
        """
        predict(file)

        Read the data from a numpy compressed file.

        Parameters
        ----------
        file : {str, Path}
            Path to the file to read. Must either be a string, or be able to be
            converted by `str(file)`.

        Returns
        -------
        data : dict
            Dictionary of the data contained in the file.

        Raises
        ------
        ValueError
            If the file name is not provided

        Notes
        -----
        The keys in `data` depend on which data the file contained. Potential keys are:

        - `accel`: acceleration [g]
        - `time`: timestamps [s]
        - `fs`: sampling frequency in Hz.
        """
        if file is None:
            raise ValueError("file must not be None")
        if not isinstance(file, str):
            file = str(file)
        if file[-3:] != "npz":
            warn("File extension is not expected '.npz'", UserWarning)
        if Path(file).stat().st_size < 1000:
            raise FileSizeError("File is less than 1kb, nothing to read.")

        data = load(file)

        kwargs.update(
            {
                self._time: data['time'],
                self._acc: data['accel'],
                'file': file
            }
        )
        if 'fs' in data:
            kwargs['fs'] = data['fs'][()]

        return (kwargs, None) if self._in_pipeline else kwargs
