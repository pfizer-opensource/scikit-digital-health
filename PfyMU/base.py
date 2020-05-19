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

    def apply(self, time, accel, angvel, temperature):
        pass