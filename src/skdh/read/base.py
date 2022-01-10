"""
Base process for file IO

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""
from pathlib import Path
import functools
from warnings import warn


def check_input_file(func):
    @functools.wraps(func)
    def _check_input_file(self, file=None, **kwargs):
        # check if the file is provided
        if file is None:
            raise ValueError("`file` must not be None.")

        # make sure the file exists
        if not Path(file).exists():
            raise FileNotFoundError(f"File {file} does not exist.")

        # check that the file matches the expected extension
        if Path(file).suffix != self.extn:
            if self.ext_error == 'warn':
                warn(self.extn_message.format(self.extn), UserWarning)
            elif self.ext_error == 'raise':
                raise ValueError(self.extn_message.format(self.extn))
            elif self.ext_error == 'skip':
                kwargs.update({'file': file})
                return (kwargs, None) if self._in_pipeline else kwargs

        return func(self, file=file, **kwargs)

    return _check_input_file
