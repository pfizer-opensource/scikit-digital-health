"""
Sleep predictions

Yiorgos Christakis, Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""

from numpy import array, convolve, int_

from skdh.sleep.utility import rle


def compute_sleep_predictions(act_index, sf=0.243, rescore=True):
    """
    Apply the Cole-Kripke algorithm to activity index data

    Parameters
    ----------
    act_index : numpy.ndarray
        Activity index calculated from accelerometer data on 1 minute windows.
    sf : float, optional
        Scale factor used for the predictions. Default is 0.243, which was optimized
        for activity index. Recommended range if changing is between 0.15 and 0.3 depending
        on desired sensitivity, and possibly the population being observed.
    rescore : bool, optional
        If True, applies Webster's rescoring rules to the sleep predictions to improve
        specificity.

    Returns
    -------

    Notes
    -----
    Applies Webster's rescoring rules as described in the Cole-Kripke paper.
    """
    # paper writes this backwards [::-1]. For convolution has to be written this way though
    kernel = array([0.0, 0.0, 4.024, 5.84, 16.19, 5.07, 3.75, 6.87, 4.64]) * sf

    scores = convolve(act_index, kernel, "same")
    predictions = (scores < 0.5).astype(int_)  # sleep as positive

    if rescore:
        wake_bin = 0
        for t in range(predictions.size):
            if not predictions[t]:
                wake_bin += 1
            else:
                if (
                    wake_bin >= 15
                ):  # rule c: >= 15 minutes of wake -> next 4min of sleep rescored
                    predictions[t : t + 4] = 0
                elif (
                    10 <= wake_bin < 15
                ):  # rule b: >= 10 minutes of wake -> next 3 min rescored
                    predictions[t : t + 3] = 0
                elif (
                    4 <= wake_bin < 10
                ):  # rule a: >=4 min of wake -> next 1min of sleep rescored
                    predictions[t] = 0
                wake_bin = 0  # reset
        # rule d: [>10 min wake][<=6 min sleep][>10min wake] gets rescored
        dt, changes, vals = rle(predictions)

        mask = (changes >= 10) & (changes < (predictions.size - 10)) & (dt <= 6) & vals
        for start, dur in zip(changes[mask], dt[mask]):
            predictions[start : start + dur] = 0
    return predictions
