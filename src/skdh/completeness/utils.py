import warnings
import os

import pandas as pd
import numpy as np



def clean_df(df, col, val_min, val_max):
    df_cleaned = df.iloc[np.where(np.array([val_min <= df[col], df[col] <= val_max]).all(axis=0))[0]]
    if len(df_cleaned) == 0:
        warnings.warn(col + ' had only one or no valid values. Removing from analysis.')
    return df_cleaned


def check_hyperparameters_init(ranges, data_gaps, time_periods, timescales):

    if ranges is not None:
        assert type(ranges) == dict, 'ranges has to be a dictionary'
        for key in ranges.keys():
            assert ranges[key][1] > ranges[key][0], 'The second and first value in ranges are the upper and lower bounds, '+\
            'respectively, so the upper value must therefore be larger than the first. This was not the case for ' + \
                                                    str(key)
    if data_gaps is not None:
        assert type(data_gaps) == np.ndarray and data_gaps.dtype in ['timedelta64[s]', 'timedelta64[m]', 'timedelta64[h]'], \
            'data_gaps must be a numpy array of dtype timedelta64'
        assert len(np.unique(data_gaps)) == len(data_gaps), 'Each element in data_gaps must be unique'
    if timescales is not None:
        assert type(timescales) == np.ndarray and timescales.dtype in ['timedelta64[s]', 'timedelta64[m]', 'timedelta64[h]'], \
            'timescales must be a numpy array of dtype timedelta64'
        assert len(np.unique(timescales)) == len(timescales), 'Each element in timescales must be unique'
    if time_periods is not None and not time_periods == 'daily':
        assert type(time_periods) == list, 'time_periods must be a list'
        for time_period in time_periods:
            assert type(time_period) == tuple and len(time_period) == 2 and type(time_period[0]) == pd.Timestamp and \
                   type(time_period[1]) == pd.Timestamp, 'each element in time_periods must be a tuple of two elements, ' +\
                                                         'both being pd.Timestamp types'

    print('All initial hyperparameters passed input controls.')

    return


def check_hyperparameters_load(subject_folder, subject, measures):

    assert type(subject) == str, 'subject parameter should be a string, identifying the subject'
    assert os.path.isdir(subject_folder), 'subject_folder does not appear to exist'
    for measure in measures:
        assert os.path.isfile(subject_folder + measure + '.csv'), 'The file ' + subject_folder + measure + '.csv' +\
                                                                  'could not be found'

    print('All loading hyperparameters passed input controls.')

    return


def check_hyperparameters_figures(resample_width_mins, gap_size_mins):

    try:
        import plotly
    except ImportError:
        raise ImportError("plotly is required for generating figures in this module")

    assert type(resample_width_mins) in [int, float], str(resample_width_mins) + ' parameter has to be an int or float'
    assert type(gap_size_mins) in [int, float], str(gap_size_mins) + ' parameter has to be an int or float'
    assert gap_size_mins >= resample_width_mins, 'gap_size_mins should be greater or equal to resample_width_mins'

    print('All figure hyperparameters passed input controls.')

    return


def find_time_periods_overlap(periods, time_segment):
    """
    Find overlap between periods (an array of time periods) and time_segment (one time period). Returns an array of
    periods that overlap. If one period is partially inside time_segment, the portion inside will be added, so that
    all overlap time will be inside time_segment. Periods have to be sorted in time and non-overlapping.
    :param periods : np.array
    :param time_segment : list with len(list) = 2
    :return: period_overlap : np.array
    """
    period_overlap = np.array([], dtype=np.timedelta64).reshape(0, 2)
    period_inds = np.array([])
    if not len(periods) == 0 and not len(time_segment) == 0:
        periods = np.array(periods, dtype=np.datetime64)
        assert np.array(np.diff(periods[:, 0]) > np.timedelta64(0)).all(), 'Periods have to be sorted in time'
        assert np.array(periods[1:, 0] - periods[:-1, 1] >= np.timedelta64(0)).all(), 'Periods have to be non-overlapping'
        time_segment = (pd.Timestamp.to_numpy(time_segment[0]), pd.Timestamp.to_numpy(time_segment[1]))
        if np.array([periods[:, 0] <= time_segment[0], periods[:, 1] >= time_segment[1]]).all(axis=0).any():
            period_overlap = np.array([time_segment])
            period_inds = np.where(np.array([periods[:, 0] <= time_segment[0],
                                             periods[:, 1] >= time_segment[1]]).all(axis=0))[0]
        else:
            obs_periods_fully_inside = np.where(np.array([periods[:, 0] >= time_segment[0],
                                                          periods[:, 1] <= time_segment[1]]).all(axis=0))[0]
            obs_periods_beg = np.where(np.array([periods[:, 0] < time_segment[0],
                                                 periods[:, 1] > time_segment[0]]).all(axis=0))[0]
            obs_periods_end = np.where(np.array([periods[:, 0] < time_segment[1],
                                                 periods[:, 1] > time_segment[1]]).all(axis=0))[0]
            period_overlap = periods[obs_periods_fully_inside]
            period_inds = np.concatenate((obs_periods_beg, obs_periods_fully_inside, obs_periods_end))
            if len(obs_periods_beg) == 1:
                period_overlap = np.concatenate(([[time_segment[0], periods[:, 1][obs_periods_beg][0]]], period_overlap))
            if len(obs_periods_end) == 1:
                period_overlap = np.concatenate((period_overlap, [[periods[:, 0][obs_periods_end][0], time_segment[1]]]))

    return period_overlap, period_inds


def find_time_periods_overlap_fraction(periods, time_segment, weights=None):
    """
    Find overlap fraction between periods (np.array) and time_segment (single list with 2 elements). Overlap fraction
    weighted by weights, which if given have to be the same size as periods. If not given, the overlap fraction will
    be a straight quota.
    :param periods:
    :param time_segment:
    :param weights:
    :return:
    """
    if weights is None:
        weights = np.ones(len(periods))
    assert len(weights) == len(periods), 'weights and periods have to be equal size'
    if np.array([periods[:, 0] <= time_segment[0], periods[:, 1] >= time_segment[1]]).all(axis=0).any():
        return np.sum(weights[np.where(np.array([periods[:, 0] <= time_segment[0], periods[:, 1] >= time_segment[1]]).all(axis=0))[0]])
    else:
        period_overlap, period_inds = find_time_periods_overlap(periods, time_segment)
        return np.sum((period_overlap[:, 1] - period_overlap[:, 0]) * weights[period_inds]) / (time_segment[1] - time_segment[0])


