"""
Function for getting the initial gait metrics

Lukas Adamowicz
Pfizer DMTI 2020
"""
from numpy import nan, cov, std
from scipy.signal import detrend
from scipy.integrate import cumtrapz


def _autocov(x, i1, i2, i3):
    ac = cov(x[i1:i2], x[i2:i3], bias=False)[0, 1]
    return ac / (std(x[i1:i2], ddof=1) * std(x[i2:i3], ddof=1))


def get_bout_metrics_delta_h(
        gait, gait_index, bout_n, dt, time, vert_accel, bout_n_steps, bout_ends,
        bout_start
):
    """
    Get the intial gait metrics obtained for each bout

    Parameters
    ----------
    gait : dictionary
        Dictionary of gait events and results. Modified in place
    gait_index : int
        Last bout position in lists of gait
    bout_n : int
        Bout number
    dt : float
        Sampling period
    time : numpy.ndarray
        (M, ) Unix timestamps
    vert_accel : numpy.ndarray
        (N, ) array of vertial acceleration
    bout_n_steps : int
        Number of steps in the current bout
    bout_ends : tuple
        Tuple of the start and stop index of the bout
    bout_start : int
        Index of the start of the bout
    """
    # get the change in height
    for i in range(bout_n_steps - 1):
        i1 = gait['IC'][gait_index + i]
        i2 = gait['IC'][gait_index + i + 1]

        if gait['b valid cycle'][gait_index + i]:
            vacc = detrend(vert_accel[i1:i2])
            vvel = cumtrapz(vacc, dx=dt, initial=0)
            vpos = cumtrapz(vvel, dx=dt, initial=0)

            gait['delta h'].append((vpos.max() - vpos.min()) * 9.81)
        else:
            gait['delta h'].append(nan)

    # make sure parameters here match the number of steps in gait
    gait['delta h'].append(nan)

    # add bout level metrics
    gait['Bout N'].extend([bout_n + 1] * bout_n_steps)
    gait['Bout Start'].extend([time[bout_start]] * bout_n_steps)
    gait['Bout Duration'].extend([(bout_ends[1] - bout_ends[0]) * dt] * bout_n_steps)
    gait['Bout Steps'].extend([sum(gait['b valid cycle'][gait_index:])] * bout_n_steps)