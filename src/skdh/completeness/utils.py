import re
import warnings
import copy
import pandas as pd
import numpy as np
from skdh.completeness.parse import vivalink_parse_ecg_data, vivalink_parse_acc_data, empatica_parse_acc_data
from skdh.completeness.helpers import convert_sfreq_to_sampling_interval


def compute_summary_metrics(completeness_master_dic, time_periods, timescales, measures):
    dic_summary = {}
    for period in time_periods:
        period_summary = []
        if completeness_master_dic['charging']['all'] is None:
            charging_time = None
        else:
            charging_time = np.sum(completeness_master_dic['charging'][period][:, 1] -
                                   completeness_master_dic['charging'][period][:, 0])
        period_summary.append(['charge time', charging_time])
        wear_time = np.sum(completeness_master_dic['wearing'][period]['wear_times'][:, 1] -
                           completeness_master_dic['wearing'][period]['wear_times'][:, 0])
        period_summary.append(['wear time', wear_time])
        no_wear_time = np.sum(completeness_master_dic['wearing'][period]['no_wear_times'][:, 1] -
                           completeness_master_dic['wearing'][period]['no_wear_times'][:, 0])
        period_summary.append(['no-wear time', no_wear_time])
        unknown_wear_time = np.sum(completeness_master_dic['wearing'][period]['unknown_times'][:, 1] -
                           completeness_master_dic['wearing'][period]['unknown_times'][:, 0])
        period_summary.append(['unknown wear time', unknown_wear_time])
        for measure in measures:
            if not measure in list(completeness_master_dic.keys()):
                period_summary.append([measure + ' completeness: 0 (no valid values)', 0])
            else:
                for key in completeness_master_dic[measure]['completeness']['native'][period].items():
                    period_summary.append([measure + ', ' + key[0] + ', native', key[1]])
                if not timescales is None:
                    for timescale in timescales:
                        for key in completeness_master_dic[measure]['completeness'][timescale][period].items():
                            period_summary.append([measure + ', ' + str(key[0]) + ', ' + str(timescale), key[1]])
                if 'data_gaps' in completeness_master_dic[measure].keys():
                    for data_gap in completeness_master_dic[measure]['data_gaps'][period].items():
                        for reason in data_gap[1].items():
                            period_summary.append([measure + ', data gap ' + str(data_gap[0]) + ', reason: ' + reason[0], reason[1]])
        period_summary = np.array(period_summary)
        dic_summary.update({period : period_summary[:, 1]})
    df_summary = pd.DataFrame(dic_summary, index=period_summary[:, 0])

    return df_summary


def dic_to_str(dic):
    if isinstance(dic, dict):
        keys = list(dic.keys())
        for key in keys:
            if not isinstance(key, str):
                dic[str(key)] = dic.pop(key)
            dic[str(key)] = dic_to_str(dic[str(key)])
    elif type(dic) == np.ndarray:
        if dic.dtype == np.dtype('O') or dic.dtype == np.dtype('<m8'):
            dic = dic.astype(str)

    return dic


def input_data_checks(df_raw, device_name):
    if device_name == 'Vivalink':
        acc_samp = []
        for k in range(len(df_raw)):
            sensor_info = df_raw['Sensor Info'].iloc[k].split(';')
            acc_samp.append(float(
                re.sub("[^0-9]", "", sensor_info[np.where(['accSamplingFrequency' in x for x in sensor_info])[0][0]])))
        assert len(np.unique(acc_samp)) == 1, 'Acc sampling not consistent.'
        assert acc_samp[0] == 5, 'Acceleration sampling not set correctly.'

    return


def init_data_dic(df_raw, columns, measures, device_name, ranges={}, ecg_key=None, acc_raw_key=None, acc_raw_fdir=None):
    assert all([measure in [x['new_name'] for x in list(columns.values())] for measure in measures if measure not in
                ['acc_raw_x', 'acc_raw_y', 'acc_raw_z']]), 'All measures have to be cols in columns'
    df = copy.deepcopy(df_raw)
    _check_columns(df, columns)
    columns_adj = copy.deepcopy(columns)
    for x in columns_adj.values():
        if type(x['sfreq']) in [int, float]: x['sfreq'] = convert_sfreq_to_sampling_interval(x['sfreq'])
        else:
            for c in range(len(df[x['sfreq']])): df.iloc[c, df.columns.get_loc(x['sfreq'])] = \
                convert_sfreq_to_sampling_interval(df.iloc[c, df.columns.get_loc(x['sfreq'])])

    columns_naming = {key : columns_adj[key]['new_name'] for key in columns_adj.keys()}
    for x in columns_adj.values():
        if isinstance(x['sfreq'], str): columns_naming.update({x['sfreq'] : 'Sampling_Freq'})

    data_dic = {'Subject ID' : df['Subject ID'].iloc[0],
                'Device Name' : device_name,
                'Measurement Streams' : {}}

    if device_name == 'VivaLink':
        if not ecg_key is None:
            df_ecg = vivalink_parse_ecg_data(df_raw, ecg_key=ecg_key, s_freq=columns['ECG']['sfreq'])
            data_dic['Measurement Streams'].update({columns_adj[ecg_key]['new_name'] : df_ecg})
        if not acc_raw_key is None:
            df_acces = vivalink_parse_acc_data(df_raw, acc_key=acc_raw_key, n_samples=columns['ACC']['sfreq'])
            for c, axis in enumerate(['x', 'y', 'z']):
                data_dic['Measurement Streams'].update({'acc_raw_' + axis : df_acces[c]})

    if device_name == 'Empatica':
        assert type(acc_raw_fdir) == str, 'For Empatica, the file path to raw data must be given in acc_raw_fdir'
        df_acces = empatica_parse_acc_data(acc_raw_fdir)
        for c, axis in enumerate(['x', 'y', 'z']):
            data_dic['Measurement Streams'].update({'acc_raw_' + axis : df_acces[c]})
        df['wearing_detection_percentage'].iloc[np.where(df['wearing_detection_percentage'] >= 75)[0]] = 1

    for col in columns_adj.keys():
        if columns_adj[col]['new_name'] not in ['ecg_raw', 'acc_raw']:
            if isinstance(columns_adj[col]['sfreq'], str):
                new_df = df[[col, 'Device ID', columns_adj[col]['sfreq']]].rename(columns=columns_naming)
            else:
                new_df = df.assign(Sampling_Freq = columns_adj[col]['sfreq'])[
                            [col, 'Device ID', 'Sampling_Freq']].rename(columns=columns_naming)
            if columns[col]['new_name'] in ranges.keys():
                new_df = clean_df(new_df, columns[col]['new_name'], ranges[columns[col]['new_name']][0], ranges[columns[col]['new_name']][1])
            if columns[col]['new_name'] in measures and len(new_df) > 0:
                data_dic['Measurement Streams'].update({columns_naming[col] : new_df})
            elif not columns[col]['new_name'] in measures:
                data_dic.update({columns_naming[col] : new_df})

    for measure in data_dic['Measurement Streams'].keys():
        data_dic['Measurement Streams'][measure] = data_dic['Measurement Streams'][measure].iloc[np.where(~np.isnan(
            data_dic['Measurement Streams'][measure][measure]))[0]]

    data_dic['Wear Indicator'] = data_dic['Wear Indicator'].iloc[np.where(~np.isnan(data_dic['Wear Indicator']['Wear Indicator']))[0]]

    return data_dic


def clean_df(df, col, val_min, val_max):
    df_cleaned = df.iloc[np.where(np.array([val_min <= df[col], df[col] <= val_max]).all(axis=0))[0]]
    if len(df_cleaned) == 0:
        warnings.warn(col + ' did not have a single valid value. Removing from analysis.')
    return df_cleaned


def _check_columns(df_raw, columns):
    assert all([x in df_raw.keys() for x in columns.keys()]), 'Some keys in columns were not in df_raw! Check columns.'
    assert all([type(x['sfreq']) in [float, int, str] for x in columns.values()]), 'sfreq must be a number (int/float) or a string (column name in df_raw containing sfreq values)'
    assert all([x['sfreq'] in df_raw.keys() for x in columns.values() if isinstance(x['sfreq'], str)]), 'sfreq must be a column in df_raw if it is a string'
    assert all([1 / x['sfreq'] * 10 ** 9 == int(1 / x['sfreq'] * 10 ** 9) for x in columns.values() if type(x['sfreq']) in [int, float]]), 'sampling interval not a nanosecond integer.'

    return


def compute_completeness_master(data_dic, data_gaps=None, time_periods=None, timescales=None):
    for time_period in time_periods:
        assert isinstance(time_period, tuple) and len(time_period) == 2 and \
               isinstance(time_period[0], pd._libs.tslibs.timestamps.Timestamp) \
            and isinstance(time_period[1], pd._libs.tslibs.timestamps.Timestamp) \
            and time_period[1] > time_period[0], \
            'time_period has to be a tuple with two date-time elements where time_period[1] is after time_period[0]'
    if not data_gaps is None:
        assert isinstance(data_gaps, np.ndarray) and \
               np.array([isinstance(data_gap, np.timedelta64) for data_gap in data_gaps]).all(), \
            'data_gaps has to be an array of np.timedelta64 elements'
    if not timescales is None:
        assert isinstance(timescales, np.ndarray) and \
               np.array([isinstance(timescale, np.timedelta64) for timescale in timescales]).all(), \
            'timescales has to be an array of np.timedelta64 elements'

    completeness_master_dic = {}

    if 'Device Battery' in list(data_dic.keys()):
        completeness_master_dic.update({'charging' : {'all' : find_charging_periods(data_dic)}})
        for time_period in time_periods:
            completeness_master_dic['charging'].update(
                {time_period: _find_time_periods_overlap(completeness_master_dic['charging']['all'], time_period)})
    else:
        completeness_master_dic.update({'charging': {'all': None}})

    if 'Wear Indicator' in list(data_dic.keys()):
        completeness_master_dic.update({'wearing': {'all': find_wear_periods(data_dic)}})
        for time_period in time_periods:
            completeness_master_dic['wearing'].update({time_period: {
                key: _find_time_periods_overlap(completeness_master_dic['wearing']['all'][key], time_period) for key in
                completeness_master_dic['wearing']['all'].keys()}})
    else:
        completeness_master_dic.update({'wearing': {'all': None}})

    for measure in data_dic['Measurement Streams'].keys():
        deltas = _find_gap_codes(data_dic, measure, completeness_master_dic['charging']['all'],
                                 completeness_master_dic['wearing']['all'])
        completeness_master_dic.update({measure : {'completeness' :
                                                       compute_completeness(deltas, time_periods=time_periods,
                                                                            timescales=timescales,
                                                                            last_time=data_dic['Measurement Streams'][
                                                                                measure].index[-1])}})
        if not data_gaps is None:
            completeness_master_dic[measure].update({'data_gaps' : _compute_data_gaps(deltas, time_periods, data_gaps)})

    return completeness_master_dic


def find_charging_periods(data_dic):
    battery_diff = np.diff(data_dic['Device Battery']['Device Battery'])
    noise_lvl = 3 * np.nanstd(battery_diff)
    charging_starts = np.where(battery_diff > noise_lvl)[0]
    charging_periods = np.array([[data_dic['Device Battery'].index[start], data_dic['Device Battery'].index[start + 1]]
                                 for start in charging_starts])

    return charging_periods


def find_wear_periods(data_dic):
    wear_indicator = np.array([np.nan] * (len(data_dic['Wear Indicator']['Wear Indicator']) - 1))
    time_periods = np.array([[data_dic['Wear Indicator'].index[c], data_dic['Wear Indicator'].index[c + 1]] for c in range(len(data_dic['Wear Indicator']) - 1)])
    for c in range(len(time_periods)):
        if (time_periods[c][1] - time_periods[c][0]) <= data_dic['Wear Indicator']['Sampling_Freq'].iloc[1:].iloc[c]:
            if data_dic['Wear Indicator']['Wear Indicator'].iloc[c] == data_dic['Wear Indicator']['Wear Indicator'].iloc[c + 1]:
                wear_indicator[c] = data_dic['Wear Indicator']['Wear Indicator'].iloc[c]
    wear_times = []
    no_wear_times = []
    unknown_times = []
    if wear_indicator[0] == 1:
        wear_times.append([time_periods[0][0], -1])
        current_ind = 1
    if wear_indicator[0] == 0:
        no_wear_times.append([time_periods[0][0], -1])
        current_ind = 0
    if np.isnan(wear_indicator[0]):
        unknown_times.append([time_periods[0][0], -1])
        current_ind = np.nan
    for c, time_period in enumerate(time_periods):
        if not wear_indicator[c] == current_ind:
            if current_ind == 1:
                wear_times[-1][1] = time_periods[c - 1][1]
            if current_ind == 0:
                no_wear_times[-1][1] = time_periods[c - 1][1]
            if np.isnan(current_ind):
                unknown_times[-1][1] = time_periods[c - 1][1]
            current_ind = wear_indicator[c]
            if wear_indicator[c] == 1:
                wear_times.append([time_period[0], -1])
                current_ind = 1
            if wear_indicator[c] == 0:
                no_wear_times.append([time_period[0], -1])
                current_ind = 0
            if np.isnan(wear_indicator[c]):
                unknown_times.append([time_period[0], -1])
                current_ind = np.nan
    if wear_indicator[-1] == 1:
        wear_times[-1][1] = time_periods[-1][1]
    if wear_indicator[-1] == 0:
        no_wear_times[-1][1] = time_periods[-1][1]
    if np.isnan(wear_indicator[-1]):
        unknown_times[-1][1] = time_periods[-1][1]
    wear_times = np.array(wear_times)
    no_wear_times = np.array(no_wear_times)
    unknown_times = np.array(unknown_times)

    return {'wear_times' : wear_times, 'no_wear_times' : no_wear_times, 'unknown_times' : unknown_times}


def _find_gap_codes(data_dic, measure, charging_periods, wearing_periods):
    dts = np.diff(data_dic['Measurement Streams'][measure].index)
    gap_codes = np.array(['unknown'] * len(dts), dtype=str)
    gap_codes = pd.DataFrame(data={'gap_codes' : gap_codes, 'dts' : dts, 'Sampling_Freq' :
        data_dic['Measurement Streams'][measure]['Sampling_Freq'][:-1]} ,
                             index=data_dic['Measurement Streams'][measure].index[:-1])
    if 'Device Battery' in list(data_dic.keys()):
        for charging_period in charging_periods:
            gap_codes.iloc[
                np.where(np.array([gap_codes.index >= charging_period[0],
                                   gap_codes.index < charging_period[1]]).all(axis=1))[
                    0], gap_codes.columns.get_loc('gap_codes')] = 'charging'
    if 'Wear Indicator' in list(data_dic.keys()):
        for no_wear_period in wearing_periods['no_wear_times']:
            gap_codes.iloc[
                np.where(np.array([gap_codes.index >= no_wear_period[0],
                                   gap_codes.index < no_wear_period[1]]).all(axis=0))[0],
                gap_codes.columns.get_loc('gap_codes')] = 'no_wear'
    gap_codes.iloc[np.where(data_dic['Measurement Streams'][measure]['Sampling_Freq'][:-1] == dts)[0],
                   gap_codes.columns.get_loc('gap_codes')] = 'normal'
    gap_codes['Sampling_Freq'] = np.array(data_dic['Measurement Streams'][measure]['Sampling_Freq'][:-1])
    gap_codes.measure = measure

    return gap_codes


def _find_time_periods_overlap(periods, time_segment):
    time_segment = (pd.Timestamp.to_numpy(time_segment[0]), pd.Timestamp.to_numpy(time_segment[1]))
    periods = np.array(periods, dtype=np.datetime64)
    if len(periods) == 0:
        return np.array([], dtype=np.timedelta64).reshape(0, 2)
    if np.array([periods[:, 0] <= time_segment[0], periods[:, 1] >= time_segment[1]]).all(axis=0).any():
        period_overlap = np.array([time_segment])
    else:
        obs_periods_fully_inside = np.where(np.array([periods[:, 0] >= time_segment[0],
                                                      periods[:, 1] <= time_segment[1]]).all(axis=0))[0]
        obs_periods_beg = np.where(np.array([periods[:, 0] < time_segment[0],
                                             periods[:, 1] > time_segment[0]]).all(axis=0))[0]
        obs_periods_end = np.where(np.array([periods[:, 0] < time_segment[1],
                                             periods[:, 1] > time_segment[1]]).all(axis=0))[0]
        period_overlap = periods[obs_periods_fully_inside]
        if len(obs_periods_beg) == 1:
            period_overlap = np.concatenate(([[time_segment[0], periods[:, 1][obs_periods_beg][0]]], period_overlap))
        if len(obs_periods_end) == 1:
            period_overlap =  np.concatenate((period_overlap, [[periods[:, 0][obs_periods_end][0], time_segment[1]]]))

    return period_overlap


def _find_time_periods_overlap_fraction(periods, time_segment):
    if np.array([periods[:, 0] <= time_segment[0], periods[:, 1] >= time_segment[1]]).all(axis=0).any():
        return 1
    else:
        period_overlap = _find_time_periods_overlap(periods, time_segment)
        return np.sum(period_overlap[:, 1] - period_overlap[:, 0]) / (time_segment[1] - time_segment[0])


def calculate_completeness_timescale(deltas, time_periods, timescale, last_time):
    assert type(timescale) in [np.timedelta64, pd.core.series.Series, np.ndarray], \
        'type(timescale) must be np.timedelta64, pd.core.series.Series or np.ndarray'
    assert type(deltas) == pd.core.frame.DataFrame and 'dts' in deltas.keys(), 'deltas must be a Pandas dataframe'
    if type(timescale) == np.timedelta64:
        timescale = np.array([timescale] * len(deltas))
    dts_inds = np.where(deltas['dts'] > timescale)[0]
    if len(dts_inds) == 0:
        observation_periods = np.array([[deltas.index[0], last_time]])
    else:
        observation_periods = [[deltas.index[0], deltas.index[dts_inds[0]] + timescale[dts_inds[0]] / 2]]
        for c in range(len(dts_inds) - 1):
            observation_periods.append([deltas.index[dts_inds[c] + 1] - timescale[dts_inds[c] + 1] / 2,
                                        deltas.index[dts_inds[c + 1]] + timescale[dts_inds[c + 1]] / 2])
        if dts_inds[-1] == len(deltas) - 1:
            observation_periods.append([last_time - timescale[dts_inds[-1]] / 2, last_time])
        else:
            observation_periods.append([deltas.index[dts_inds[-1] + 1] - timescale[dts_inds[-1] + 1] / 2, last_time])
        observation_periods = np.array(observation_periods)
    reason_periods = {}
    for reason in deltas['gap_codes'][dts_inds].unique():
#        if not reason == 'normal':
        reason_inds = np.where(deltas['gap_codes'].iloc[dts_inds] == reason)[0]
        if dts_inds[reason_inds][-1] == len(deltas) - 1:
            reason_periods.update(
                {reason: np.array(
                    [deltas.index[dts_inds[reason_inds[:-1]]] + timescale[dts_inds[reason_inds[:-1]]] / 2,
                     deltas.index[dts_inds[reason_inds[:-1]] + 1] - timescale[
                         dts_inds[reason_inds[:-1]] + 1] / 2]).T.reshape(len(reason_inds[:-1]), 2)})
            reason_periods.update({reason : np.concatenate((reason_periods[reason], np.array(
                    [deltas.index[dts_inds[reason_inds[-1]]] + timescale[dts_inds[reason_inds[-1]]] / 2,
                     last_time - timescale[-1] / 2], dtype=np.datetime64).T.reshape((1, 2))), axis=0)})
        else:
            reason_periods.update({reason : np.array([deltas.index[dts_inds[reason_inds]] + timescale[dts_inds][reason_inds] / 2,
                                   deltas.index[dts_inds[reason_inds] + 1] - timescale[dts_inds[reason_inds] + 1] / 2]).T.reshape(len(reason_inds), 2)})
    data_completeness = {}
    if deltas['gap_codes'].index[0] > time_periods[0][0]:
        if 'unknown' in reason_periods.keys():
            reason_periods.update({'unknown' : np.concatenate((np.array([[time_periods[0][0],
                                                                          observation_periods[0][0]]],
                                                                        dtype=np.datetime64),
                                                               reason_periods['unknown']), axis=0)})
        else:
            reason_periods.update({'unknown' : np.array([[time_periods[0][0], observation_periods[0][0]]], dtype=np.datetime64)})
    if last_time < time_periods[-1][1]:
        last_time_point = np.min([last_time + timescale[-1] / 2, time_periods[-1][1]])
        observation_periods = np.concatenate(
            (observation_periods, np.array([[last_time, last_time_point]])))
        if 'unknown' in reason_periods.keys():
            reason_periods.update({'unknown' : np.concatenate((reason_periods['unknown'],
                                                               np.array([[last_time_point, time_periods[-1][1]]],
                                                                        dtype=np.datetime64)), axis=0)})
        else:
            reason_periods.update({'unknown' : np.array([[last_time_point, time_periods[-1][1]]],
                                                                        dtype=np.datetime64)})

    for time_period in time_periods:
        data_completeness.update({time_period: {'completeness' : _find_time_periods_overlap_fraction(observation_periods, time_period)}})
        for reason in reason_periods.keys():
            data_completeness[time_period].update({'missingness, ' + reason : _find_time_periods_overlap_fraction(reason_periods[reason], time_period)})
    if np.abs(np.sum(list(list(data_completeness.values())[0].values())) - 1) > .005:
        dsfdnsjk
    return data_completeness


def compute_completeness(deltas, time_periods, last_time, timescales=None):
    completeness = {'native' : calculate_completeness_timescale(deltas=deltas, time_periods=time_periods,
                                                                timescale=deltas['Sampling_Freq'],
                                                                last_time=last_time)}
    if not timescales is None:
        for timescale in timescales:
            completeness.update({timescale : calculate_completeness_timescale(deltas=deltas, time_periods=time_periods,
                                                                              timescale=timescale, last_time=last_time)})

    return completeness


def _compute_data_gaps(deltas, time_periods, data_gaps):
    data_gap_summary = {}
    for time_period in time_periods:
        data_gap_summary.update({time_period : {}})
        for data_gap in data_gaps:
            data_gap_inds = np.where(np.array([deltas['dts'] >= data_gap, deltas['dts'].index >= time_period[0], deltas['dts'].index < time_period[1]]).all(axis=0))[0]
            reasons = deltas['gap_codes'].unique()
            data_gap_reason = {}
            for reason in reasons:
                data_gap_reason.update(
                    {reason: np.sum(deltas.iloc[data_gap_inds, deltas.columns.get_loc('gap_codes')] == reason)})
            data_gap_summary[time_period].update({data_gap: data_gap_reason})

    return data_gap_summary


def truncate_data_dic(data_dic, time_period):
    data_dic_trunc = copy.deepcopy(data_dic)
    for stream in data_dic_trunc['Measurement Streams'].keys():
        data_dic_trunc['Measurement Streams'][stream] = data_dic_trunc['Measurement Streams'][stream].iloc[
            np.where(np.array([data_dic_trunc['Measurement Streams'][stream].index >= time_period[0],
                               data_dic_trunc['Measurement Streams'][stream].index <= time_period[1]]).all(axis=0))[0]]
    data_dic_trunc['Wear Indicator'] = data_dic_trunc['Wear Indicator'].iloc[
        np.where(np.array([data_dic_trunc['Wear Indicator'].index >= time_period[0],
                           data_dic_trunc['Wear Indicator'].index <= time_period[1]]).all(axis=0))[0]]
    data_dic_trunc['Device Battery'] = data_dic_trunc['Device Battery'].iloc[
        np.where(np.array([data_dic_trunc['Device Battery'].index >= time_period[0],
                           data_dic_trunc['Device Battery'].index <= time_period[1]]).all(axis=0))[0]]

    return data_dic_trunc


