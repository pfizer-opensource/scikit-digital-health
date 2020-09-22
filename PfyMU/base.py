"""
Base classes, functions, etc for the PfyMU library

Lukas Adamowicz
Pfizer DMTI 2020
"""


class _BaseProcess:
    def __init__(self):
        """
        Intended to be subclassed
        """
        pass

    def predict(self, file=None, time=None, accel=None, angvel=None, temperature=None, **kwargs):
        pass

    def _predict(self, *, file=None, time=None, accel=None, angvel=None, temperature=None, **kwargs):
        pass
