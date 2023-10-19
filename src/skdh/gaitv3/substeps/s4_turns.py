"""
Functions for getting turns.

Lukas Adamowicz
Copyright 2023 Pfizer Inc, all rights reserved
"""
from numpy import full, int_

from skdh.base import BaseProcess


class TurnDetection(BaseProcess):
    """
    Get the location of turns, to indicate if steps occur during a turn.

    Notes
    -----
    Values indicate turns as follows:

    - -1: Turns not detected (lacking angular velocity data)
    - 0: No turn found
    - 1: Turn overlaps with either Initial or Final contact
    - 2: Turn overlaps with both Initial and Final contact

    References
    ----------
    .. [1] M. H. Pham et al., “Algorithm for Turning Detection and Analysis
        Validated under Home-Like Conditions in Patients with Parkinson’s Disease
        and Older Adults using a 6 Degree-of-Freedom Inertial Measurement Unit at
        the Lower Back,” Front. Neurol., vol. 8, Apr. 2017,
        doi: 10.3389/fneur.2017.00135.
    """
    def __init__(self):
        super().__init__()

    def predict(self, time=None, accel=None, gyro=None, qc_initial_contacts=None, qc_final_contacts=None, *, fs=None, **kwargs):
        """
        predict(time, accel, gyro, qc_initial_contacts, qc_final_contacts, *, fs=None)

        Parameters
        ----------
        time
        accel
        gyro
        qc_initial_contacts
        qc_final_contacts
        fs

        Returns
        -------

        """
        # get the number of strides
        n_steps = qc_initial_contacts.size
        turns = full(n_steps, -1, dtype=int_)  # allocate

        # check if we can detect turns
        if gyro is None:
            pass
