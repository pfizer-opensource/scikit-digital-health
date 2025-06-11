import os
import warnings
import pickle
import copy

import numpy as np
import pandas as pd

from skdh.base import BaseProcess, handle_process_returns
from skdh.completeness.helpers import from_unix
from skdh.completeness.utils import check_hyperparameters_init, check_hyperparameters_load, check_hyperparameters_figures, clean_df, find_time_periods_overlap, find_time_periods_overlap_fraction
from skdh.completeness.visualizations import visualize_overview_plot, plot_completeness, plot_data_gaps, plot_timescale_completeness



class AssessCompleteness(BaseProcess):
    r"""
    Pipeline for assessing signal completeness.
    """

    def __init__(
            self,
            ranges=None,
            data_gaps=None,
            time_periods=None,
            timescales=None):

        check_hyperparameters_init(
            ranges,
            data_gaps,
            time_periods,
            timescales)

        super().__init__(
            ranges=ranges,
            data_gaps=data_gaps,
            time_periods=time_periods,
            timescales=timescales)

        self.ranges = ranges
        self.data_gaps = data_gaps
        self.time_periods = time_periods
        self.timescales = timescales

    def load_subject_data(self, subject_folder, subject, measures):

        check_hyperparameters_load(
            subject_folder,
            subject,
            measures)

        data_dic = {'Subject ID': subject,
                    'Measurement Streams': {}}

        data_files = measures
        if 'Charging Indicator.csv' in os.listdir(subject_folder):
            data_files = data_files + ['Charging Indicator']
        if 'Wear Indicator.csv' in os.listdir(subject_folder):
            data_files = data_files + ['Wear Indicator']

        for measure in data_files:
            fname = subject_folder + '/' + measure + '.csv'
            df_raw = pd.read_csv(fname)

            assert 'Time Unix (ms)' in df_raw.keys(), '"Time Unix (ms)" column is missing from file ' + fname
            assert 'Sampling Frequency (Hz)' in df_raw.keys(), '"Sampling Frequency (Hz)" column is missing from file ' + fname
            assert measure in df_raw.keys(), measure + ' column is missing from file ' + fname
            assert df_raw['Time Unix (ms)'].iloc[0] < 10 ** 13 and df_raw['Time Unix (ms)'].iloc[0] > 10 ** 11, \
                'Unix times are too small or too big, they need to be in ms and should therefore be ~10^12'
            assert df_raw['Sampling Frequency (Hz)'].dtype in [float, int], 'sfreq must be a number (int/float)'

            if not 'Device ID' in df_raw.keys():
                df_raw['Device ID'] = 'n/a'

            if not 'Timezone' in df_raw.keys():
                warnings.warn('No timezone key given, will presume the timezone is EST (-5).')
                times = [from_unix(df_raw['Time Unix (ms)'], time_unit='ms', utc_offset=-5)]
                df_raw.set_index(times, inplace=True)
            else:
                times = [from_unix(df_raw['Time Unix (ms)'], time_unit='ms', utc_offset=df_raw['Timezone'])]
                df_raw.set_index(times, inplace=True)

            df_raw['Sampling Frequency (Hz)'] = np.array(1 / df_raw['Sampling Frequency (Hz)'] * 10 ** 9,
                                                         'timedelta64[ns]')
            df_raw = df_raw.iloc[np.where(~np.isnan(df_raw[measure]))[0]]
            df_raw = df_raw.iloc[np.argsort(df_raw['Time Unix (ms)'])]

            if not self.ranges is None:
                if measure in self.ranges.keys():
                    df_raw = clean_df(df_raw, measure, self.ranges[measure][0], self.ranges[measure][1])
            if len(df_raw) > 1:
                if measure in ['Charging Indicator', 'Wear Indicator']:
                    data_dic.update({measure: df_raw})
                else:
                    data_dic['Measurement Streams'].update({measure: df_raw})

        return data_dic

    @handle_process_returns(results_to_kwargs=True)
    def predict(self,
                data_dic,
                measures,
                fpath_output=None,
                **kwargs):
        """
        Compute completeness and save results. Create and save figures if generate_figures is True.
        """

        assert os.path.isdir(fpath_output), 'fpath_output does not appear to exist'

        super().predict(
            expect_days=False,
            expect_wear=False,
            **kwargs,
        )

        if self.time_periods is None:
            self.time_periods = [(np.min([x.index[0] for x in data_dic['Measurement Streams'].values()]),
                                  np.max([x.index[-1] for x in data_dic['Measurement Streams'].values()]))]
        elif self.time_periods == 'daily':
            t0 = np.min([x.index[0] for x in data_dic['Measurement Streams'].values()])
            t1 = np.max([x.index[-1] for x in data_dic['Measurement Streams'].values()])
            no_days = int(np.ceil((t1 - t0) / np.timedelta64(24, 'h')))
            self.time_periods = [(t0 + k * np.timedelta64(24, 'h'), t0 + (k + 1) * np.timedelta64(24, 'h') if
                                t1 > t0 + (k + 1) * np.timedelta64(24, 'h') else t1) for k in range(no_days)]

        # Compute completeness metrics
        self.completeness_master_dic = self.compute_completeness_master(data_dic,
                                                                        data_gaps=self.data_gaps,
                                                                        time_periods=self.time_periods,
                                                                        timescales=self.timescales)

        # Save raw results and summary metrics
        if not fpath_output is None:
            pickle.dump(self.completeness_master_dic,
                        open(fpath_output + '/raw_completeness', 'wb'))
        self.df_summary = self.compute_summary_metrics(self.completeness_master_dic, self.time_periods, self.timescales,
                                                  measures)  # Daily wear time, charging, data gaps, native completeness

        ###### Vida specific metrics
        if 'Wear Indicator' in data_dic.keys():
            wear_basic = self.compute_wear_basic(data_dic, self.time_periods)
            df_wear_basic = pd.DataFrame(wear_basic)
            self.df_summary = pd.concat([df_wear_basic, self.df_summary])
        ###### Vida metrics end

        if not fpath_output is None:
            self.df_summary.to_csv(fpath_output + '/summary_metrics.csv')

        return {"Completeness": {'data_dic' : data_dic, 'compl_dic' : self.completeness_master_dic,
                                 'df_summary' : self.df_summary}}

    def compute_completeness_master(self, data_dic, data_gaps=None, time_periods=None, timescales=None):
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

        completeness_master_dic.update({'Wear': {'all': None}})
        completeness_master_dic.update({'Charging': {'all': None}})
        for x in list(data_dic.keys()):
            if x in ['Charging Indicator', 'Wear Indicator']:
                completeness_master_dic.update({x.split(' ')[0]: {'all': self.find_periods(data_dic, key=x)}})
                for time_period in time_periods:
                    completeness_master_dic[x.split(' ')[0]].update({time_period: {
                        key: find_time_periods_overlap(completeness_master_dic[x.split(' ')[0]]['all'][key], time_period)[0] for
                        key in
                        completeness_master_dic[x.split(' ')[0]]['all'].keys()}})

        for measure in data_dic['Measurement Streams'].keys():
            deltas = self.find_gap_codes(data_dic, measure, completeness_master_dic['Charging']['all'],
                                     completeness_master_dic['Wear']['all'])
            completeness_master_dic.update({measure: {'Completeness':
                                                          self.compute_completeness(deltas, time_periods=time_periods,
                                                                               timescales=timescales,
                                                                               last_time=
                                                                               data_dic['Measurement Streams'][
                                                                                   measure].index[-1])}})
            if not data_gaps is None:
                completeness_master_dic[measure].update(
                    {'data_gaps': self.compute_data_gaps(deltas, time_periods, data_gaps)})

        return completeness_master_dic


    def find_periods(self, data_dic, key):
        """

        :param data_dic:
        :param key: str 'Wear Indicator' | 'Charging Indicator'
        :return:
        """

        assert key in ['Wear Indicator', 'Charging Indicator'], "'key' needs to be either 'Wear Indicator' or 'Charging Indicator'"

        on_indicator = np.array([np.nan] * (len(data_dic[key][key]) - 1))
        time_periods = np.array([[data_dic[key].index[c], data_dic[key].index[c + 1]] for c in
                                 range(len(data_dic[key]) - 1)])
        for c in range(len(time_periods)):
            if (time_periods[c][1] - time_periods[c][0]) <= \
                    data_dic[key]['Sampling Frequency (Hz)'].iloc[1:].iloc[c]:
                if data_dic[key][key].iloc[c] == \
                        data_dic[key][key].iloc[c + 1]:
                    on_indicator[c] = data_dic[key][key].iloc[c]
        on_times = []
        no_on_times = []
        unknown_times = []
        if on_indicator[0] == 1:
            on_times.append([time_periods[0][0], -1])
            current_ind = 1
        if on_indicator[0] == 0:
            no_on_times.append([time_periods[0][0], -1])
            current_ind = 0
        if np.isnan(on_indicator[0]):
            unknown_times.append([time_periods[0][0], -1])
            current_ind = np.nan
        for c, time_period in enumerate(time_periods):
            if not on_indicator[c] == current_ind and not np.array([np.isnan(on_indicator[c]), np.isnan(current_ind)]).all():
                if current_ind == 1:
                    on_times[-1][1] = time_periods[c - 1][1]
                if current_ind == 0:
                    no_on_times[-1][1] = time_periods[c - 1][1]
                if np.isnan(current_ind):
                    unknown_times[-1][1] = time_periods[c - 1][1]
                current_ind = on_indicator[c]
                if on_indicator[c] == 1:
                    on_times.append([time_period[0], -1])
                    current_ind = 1
                if on_indicator[c] == 0:
                    no_on_times.append([time_period[0], -1])
                    current_ind = 0
                if np.isnan(on_indicator[c]):
                    unknown_times.append([time_period[0], -1])
                    current_ind = np.nan
        if on_indicator[-1] == 1:
            on_times[-1][1] = time_periods[-1][1]
        if on_indicator[-1] == 0:
            no_on_times[-1][1] = time_periods[-1][1]
        if np.isnan(on_indicator[-1]):
            unknown_times[-1][1] = time_periods[-1][1]
        on_times = np.array(on_times)
        no_on_times = np.array(no_on_times)
        unknown_times = np.array(unknown_times)
        results_dic = {key.split(' ')[0] + '_times' : on_times,
                       'no_' + key.split(' ')[0] + '_times' : no_on_times,
                       'unknown_' + key.split(' ')[0] + '_times' : unknown_times}
        return results_dic

    def compute_wear_basic(self, data_dic, time_periods):

        wear_basics = {}
        for time_period in time_periods:
            wear_basics.update({time_period: {}})
            wear_ind = data_dic['Wear Indicator'].iloc[np.array([
                data_dic['Wear Indicator'].index >= time_period[0],
                data_dic['Wear Indicator'].index < time_period[1]]).all(axis=0)]
            wear_basics[time_period].update({'wear_time_basic':
                                                 np.sum(
                                                     wear_ind['Wear Indicator'] * wear_ind['Sampling Frequency (Hz)'])})
            wear_basics[time_period].update({'non_wear_time_basic': np.sum(wear_ind['Sampling Frequency (Hz)'].iloc[
                                                                               np.where(
                                                                                   wear_ind['Wear Indicator'] == 0.)[
                                                                                   0]])})
            wear_basics[time_period].update({'nan_time_basic': np.sum(np.isnan(wear_ind['Wear Indicator']) *
                                                                      wear_ind['Sampling Frequency (Hz)'])})
            assert np.array([pd.isnull(x) for x in list(wear_basics[time_period].values())]).any() == False, 'Nan values in wear_basics'
        return wear_basics

    def find_gap_codes(self, data_dic, measure, charging_periods, wearing_periods):

        dts = np.diff(data_dic['Measurement Streams'][measure].index)
        gap_codes = pd.DataFrame(data={'unknown': np.ones(len(dts)),
                                       'normal': np.zeros(len(dts)),
                                       'charging': np.zeros(len(dts)),
                                       'no_wear': np.zeros(len(dts)),
                                       'dts': dts,
                                       'Sampling Frequency (Hz)': data_dic['Measurement Streams'][measure][
                                                                      'Sampling Frequency (Hz)'][:-1]},
                                 index=data_dic['Measurement Streams'][measure].index[:-1])
        data_gaps = np.where((dts > data_dic['Measurement Streams'][measure]['Sampling Frequency (Hz)'][:-1]))[0]
        normals = np.where(dts <= data_dic['Measurement Streams'][measure]['Sampling Frequency (Hz)'][:-1])[0]
        gap_codes.iloc[normals, gap_codes.columns.get_loc('unknown')] = 0
        gap_codes.iloc[normals, gap_codes.columns.get_loc('normal')] = 1

        # Assign gap codes based on share of data gap
        for row_ind in data_gaps:
            data_gap_start = data_dic['Measurement Streams'][measure].index[row_ind]
            data_gap_end = data_dic['Measurement Streams'][measure].index[row_ind + 1]
            data_gap_duration = data_gap_end - data_gap_start

            # Charging
            charging_time = np.timedelta64(0)
            if not charging_periods is None:
                if not charging_periods['Charging_times'] is None:
                    overlap = find_time_periods_overlap(charging_periods['Charging_times'], [data_gap_start, data_gap_end])[0]
                    charging_time = np.sum(overlap[:, 1] - overlap[:, 0])

            # Non-wearing
            nonwear_time = np.timedelta64(0)
            if not wearing_periods is None:
                if not wearing_periods['no_Wear_times'] is None:
                    overlap = \
                    find_time_periods_overlap(wearing_periods['no_Wear_times'], [data_gap_start, data_gap_end])[0]
                    nonwear_time = np.sum(overlap[:, 1] - overlap[:, 0])

            unknown_time = data_gap_duration - (nonwear_time + charging_time)

            gap_codes.iloc[row_ind, gap_codes.columns.get_loc('unknown')] = unknown_time / data_gap_duration
            gap_codes.iloc[row_ind, gap_codes.columns.get_loc('charging')] = charging_time / data_gap_duration
            gap_codes.iloc[row_ind, gap_codes.columns.get_loc('no_wear')] = nonwear_time / data_gap_duration

            assert np.sum(gap_codes.iloc[row_ind, gap_codes.columns.get_loc('unknown')] +
                          gap_codes.iloc[row_ind, gap_codes.columns.get_loc('normal')] +
                          gap_codes.iloc[row_ind, gap_codes.columns.get_loc('charging')] +
                          gap_codes.iloc[
                              row_ind, gap_codes.columns.get_loc('no_wear')]) == 1, 'Sum of data gap reasons != 1'

        gap_codes.measure = measure

        return gap_codes


    def compute_summary_metrics(self, completeness_master_dic, time_periods, timescales, measures):
        dic_summary = {}
        for period in time_periods:
            period_summary = []

            # Wearing and Charging
            for key in ['Wear', 'Charging']:
                if completeness_master_dic[key]['all'] is None:
                    wear_time = None
                else:
                    wear_time = np.sum(completeness_master_dic[key][period][key + '_times'][:, 1] -
                                       completeness_master_dic[key][period][key + '_times'][:, 0])
                    period_summary.append([key + '_times', wear_time])
                    no_wear_time = np.sum(completeness_master_dic[key][period]['no_' + key + '_times'][:, 1] -
                                          completeness_master_dic[key][period]['no_' + key + '_times'][:, 0])
                    period_summary.append(['no_' + key + '_times', no_wear_time])
                    unknown_wear_time = np.sum(completeness_master_dic[key][period]['unknown_' + key + '_times'][:, 1] -
                                               completeness_master_dic[key][period]['unknown_' + key + '_times'][:, 0])
                    period_summary.append(['unknown_' + key + '_times', unknown_wear_time])

            # Measures
            for measure in measures:
                if not measure in list(completeness_master_dic.keys()):
                    period_summary.append([measure + ' completeness: 0 (no valid values)', 0])
                else:
                    for key in completeness_master_dic[measure]['Completeness']['native'][period].items():
                        period_summary.append([measure + ', ' + key[0] + ', native', key[1]])
                    if not timescales is None:
                        for timescale in timescales:
                            for key in completeness_master_dic[measure]['Completeness'][timescale][period].items():
                                period_summary.append([measure + ', ' + str(key[0]) + ', ' + str(timescale), key[1]])
                    if 'data_gaps' in completeness_master_dic[measure].keys():
                        for data_gap in completeness_master_dic[measure]['data_gaps'][period].items():
                            for reason in data_gap[1].items():
                                period_summary.append(
                                    [measure + ', data gap ' + str(data_gap[0]) + ', reason: ' + reason[0], reason[1]])

            period_summary = np.array(period_summary)
            dic_summary.update({period: period_summary[:, 1]})
        df_summary = pd.DataFrame(dic_summary, index=period_summary[:, 0])

        return df_summary

    def calculate_completeness_timescale(self, deltas, time_periods, timescale, last_time):
        assert type(timescale) in [np.timedelta64, pd.core.series.Series, np.ndarray], \
            'type(timescale) must be np.timedelta64, pd.core.series.Series or np.ndarray'
        assert type(deltas) == pd.core.frame.DataFrame and 'dts' in deltas.keys(), 'deltas must be a Pandas dataframe'
        if type(timescale) == np.timedelta64:
            timescale = np.array([timescale] * len(deltas))
        timescale = np.array(timescale / np.timedelta64(1, 'ns'),
                             dtype='timedelta64[ns]')  # To avoid rounding errors due to int64
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
                observation_periods.append(
                    [deltas.index[dts_inds[-1] + 1] - timescale[dts_inds[-1] + 1] / 2, last_time])
            observation_periods = np.array(observation_periods)
        reason_periods = {'periods': {}, 'weights': {}}
        for reason in ['normal', 'charging', 'no_wear', 'unknown']:
            reason_inds = np.where(deltas.iloc[dts_inds][reason] > 0)[0]
            reason_periods['weights'].update({reason: np.array(deltas.iloc[dts_inds][reason])})
            if len(reason_inds) > 0:
                if dts_inds[reason_inds][-1] == len(deltas) - 1:
                    reason_periods['periods'].update(
                        {reason: np.array(
                            [deltas.index[dts_inds[reason_inds[:-1]]] + timescale[dts_inds[reason_inds[:-1]]] / 2,
                             deltas.index[dts_inds[reason_inds[:-1]] + 1] - timescale[
                                 dts_inds[reason_inds[:-1]] + 1] / 2]).T.reshape(len(reason_inds[:-1]), 2)})
                    reason_periods['periods'].update(
                        {reason: np.concatenate((reason_periods['periods'][reason], np.array(
                            [deltas.index[dts_inds[reason_inds[-1]]] + timescale[dts_inds[reason_inds[-1]]] / 2,
                             last_time - timescale[-1] / 2], dtype=np.datetime64).T.reshape((1, 2))), axis=0)})
                else:
                    reason_periods['periods'].update(
                        {reason: np.array([deltas.index[dts_inds[reason_inds]] + timescale[dts_inds][reason_inds] / 2,
                                           deltas.index[dts_inds[reason_inds] + 1] - timescale[
                                               dts_inds[reason_inds] + 1] / 2]).T.reshape(len(reason_inds), 2)})
        data_completeness = {}
        if deltas.index[0] > time_periods[0][0]:
            if 'unknown' in reason_periods['periods'].keys():
                reason_periods['periods'].update({'unknown': np.concatenate((np.array([[time_periods[0][0],
                                                                                        observation_periods[0][0]]],
                                                                                      dtype=np.datetime64),
                                                                             reason_periods['periods']['unknown']),
                                                                            axis=0)})
                reason_periods['weights'].update({'unknown': np.insert(reason_periods['weights']['unknown'], 0, 1)})
            else:
                reason_periods['periods'].update(
                    {'unknown': np.array([[time_periods[0][0], observation_periods[0][0]]], dtype=np.datetime64)})
                reason_periods['weights'].update({'unknown': np.array([1])})
        if last_time < time_periods[-1][1]:
            last_time_point = np.min([last_time + timescale[-1] / 2, time_periods[-1][1]])
            observation_periods = np.concatenate(
                (observation_periods, np.array([[last_time, last_time_point]])))
            if 'unknown' in reason_periods['periods'].keys():
                reason_periods['periods'].update({'unknown': np.concatenate((reason_periods['periods']['unknown'],
                                                                             np.array([[last_time_point,
                                                                                        time_periods[-1][1]]],
                                                                                      dtype=np.datetime64)), axis=0)})
                reason_periods['weights'].update({'unknown': np.append(reason_periods['weights']['unknown'], 1)})
            else:
                reason_periods['periods'].update({'unknown': np.array([[last_time_point, time_periods[-1][1]]],
                                                                      dtype=np.datetime64)})
                reason_periods['weights'].update({'unknown': np.array([1])})

        for time_period in time_periods:
            data_completeness.update(
                {time_period: {'Completeness': find_time_periods_overlap_fraction(observation_periods, time_period)}})
            for reason in reason_periods['periods'].keys():
                weights = reason_periods['weights'][reason][np.where(reason_periods['weights'][reason] > 0)[0]]
                data_completeness[time_period].update({'Missingness, ' + reason: find_time_periods_overlap_fraction(
                    reason_periods['periods'][reason], time_period, weights)})
        for dict_tp in data_completeness.values():
            assert np.abs(np.sum(list(dict_tp.values())) - 1) < .005, \
            'Completeness + Missingness less than 99.5% (should be 100%). Something is wrong!'

        return data_completeness

    def compute_completeness(self, deltas, time_periods, last_time, timescales=None):
        completeness = {'native': self.calculate_completeness_timescale(deltas=deltas, time_periods=time_periods,
                                                                   timescale=deltas['Sampling Frequency (Hz)'],
                                                                   last_time=last_time)}
        if not timescales is None:
            for timescale in timescales:
                completeness.update(
                    {timescale: self.calculate_completeness_timescale(deltas=deltas, time_periods=time_periods,
                                                                 timescale=timescale, last_time=last_time)})

        return completeness

    def compute_data_gaps(self, deltas, time_periods, data_gaps):

        # Assign data gaps based on majority vote
        reasons_ind = np.argmax(np.array([deltas['normal'], deltas['unknown'], deltas['no_wear'], deltas['charging']]),
                                axis=0)
        deltas['gap_codes_majority'] = np.array(['normal', 'unknown', 'no_wear', 'charging'])[reasons_ind]

        data_gap_summary = {}
        for time_period in time_periods:
            data_gap_summary.update({time_period: {}})
            for data_gap in data_gaps:
                data_gap_inds = np.where(np.array([deltas['dts'] >= data_gap, deltas['dts'].index >= time_period[0],
                                                   deltas['dts'].index < time_period[1]]).all(axis=0))[0]
                reasons = deltas['gap_codes_majority'].unique()
                data_gap_reason = {}
                for reason in reasons:
                    data_gap_reason.update(
                        {reason: np.sum(
                            deltas.iloc[data_gap_inds, deltas.columns.get_loc('gap_codes_majority')] == reason)})
                data_gap_summary[time_period].update({data_gap: data_gap_reason})

        return data_gap_summary

    def truncate_data_dic(self, data_dic, time_period):
        data_dic_trunc = copy.deepcopy(data_dic)
        for stream in data_dic_trunc['Measurement Streams'].keys():
            data_dic_trunc['Measurement Streams'][stream] = data_dic_trunc['Measurement Streams'][stream].iloc[
                np.where(np.array([data_dic_trunc['Measurement Streams'][stream].index >= time_period[0],
                                   data_dic_trunc['Measurement Streams'][stream].index <= time_period[1]]).all(axis=0))[
                    0]]
        data_dic_trunc['Wear Indicator'] = data_dic_trunc['Wear Indicator'].iloc[
            np.where(np.array([data_dic_trunc['Wear Indicator'].index >= time_period[0],
                               data_dic_trunc['Wear Indicator'].index <= time_period[1]]).all(axis=0))[0]]
        data_dic_trunc['Charging Indicator'] = data_dic_trunc['Charging Indicator'].iloc[
            np.where(np.array([data_dic_trunc['Charging Indicator'].index >= time_period[0],
                               data_dic_trunc['Charging Indicator'].index <= time_period[1]]).all(axis=0))[0]]

        return data_dic_trunc

    def generate_figures(self, fpath_output, data_dic, resample_width_mins, gap_size_mins, fontsize=None):

        check_hyperparameters_figures(resample_width_mins, gap_size_mins)
        # Create and save visualizations of results
        figures = {}
        overview_fig = visualize_overview_plot(data_dic=data_dic, fpath=fpath_output + 'overview.html',
                                               resample_width_mins=resample_width_mins, gap_size_mins=gap_size_mins,
                                               time_periods=self.time_periods, fontsize=fontsize)
        reason_color_dic = {'normal' : 'blue', 'unknown' : 'magenta', 'charging' : 'orange', 'no_wear' : 'red'}
        figures.update({'overview': overview_fig})
        completeness_fig = plot_completeness(self.completeness_master_dic, data_dic, self.time_periods,
                                             fpath=fpath_output + 'completeness', reason_color_dic=reason_color_dic)
        figures.update({'Completeness': completeness_fig})
        if not self.data_gaps is None:
            data_gap_fig = plot_data_gaps(self.completeness_master_dic, data_dic, self.data_gaps, self.time_periods,
                                          fpath=fpath_output + 'data_gaps', reason_color_dic=reason_color_dic)
            figures.update({'data_gaps': data_gap_fig})
        if not self.timescales is None:
            timescale_compl_fig = plot_timescale_completeness(self.completeness_master_dic, data_dic, self.time_periods,
                                                              self.timescales,
                                                              fpath=fpath_output + 'timescale_completeness',
                                                              reason_color_dic=reason_color_dic)
            figures.update({'timescale_completeness': timescale_compl_fig})

        return figures




