import numpy as np
import pandas as pd
import warnings
import re
import os
from skdh.completeness.helpers import from_unix, convert_sfreq_to_sampling_interval

def vivalink_parse_ecg_data(df, ecg_key, s_freq):
    if type(s_freq) == int:
        n_samples = np.repeat(s_freq, repeats=len(df))
        sfreqs = np.array((1 / n_samples * 10 ** 9).astype(int), dtype='<m8[ns]')
    elif type(s_freq) == str:
        n_samples = df[s_freq]
        sfreqs = np.array([], dtype='<m8[ns]')
        for c in range(len(df)):
            sfreqs = np.append(sfreqs, np.repeat(np.timedelta64(int(1 / df.iloc[c, df.columns.get_loc(s_freq)] * 10 ** 9), 'ns'), repeats=df[s_freq].iloc[c]))
    else:
        raise TypeError('n_samples has to be a string if a key in df, or an int')
    ecg_signal = []
    device = np.array([])
    df_ecg = pd.DataFrame()
    times = np.array([])
    for k in range(len(df)):
        ecg_records = df[ecg_key].iloc[k].replace('[', '').replace(']', '').split(',')
        ecg_segment = np.array([np.nan] * n_samples[k])
        if not len(ecg_records) == n_samples[k]:
            warnings.warn('Not correct number of samples in df entry. Presuming they are in the beginning of the'
                          ' measurement period.')
        for c, ele in enumerate(df[ecg_key].iloc[k].replace('[', '').replace(']', '').split(',')):
            if 'null' not in ele:
                ecg_segment[c] = float(ele)
            else:
                ecg_segment[c] = np.nan
        ecg_signal.append(ecg_segment)
        times = np.append(times, np.array(df['Record Time'][k]).repeat(n_samples[k]) + (np.arange(n_samples[k]) * 1 / n_samples[k] * 1000))
        device = np.append(device, np.array(df['Device ID'][k]).repeat(n_samples[k]))
    ecg_signal = np.array(ecg_signal).ravel()
    df_ecg['Time Unix (ms)'] = times
    df_ecg['ecg_raw'] = ecg_signal
    df_ecg['Sampling_Freq'] = sfreqs
    df_ecg['Device ID'] = device

    return df_ecg


def vivalink_parse_acc_data(df, acc_key):

    acc_samp = []
    for k in range(len(df)):
        sensor_info = df['Sensor Info'].iloc[k].split(';')
        acc_samp.append(int(
            re.sub("[^0-9]", "", sensor_info[np.where(['accSamplingFrequency' in x for x in sensor_info])[0][0]])))

    acc_signal = np.array([]).reshape((0, 3))
    device = np.array([])
    times = np.array([])
    sfreq = np.array([])
    for k in range(len(df)):
        acc_records = np.array(df[acc_key].iloc[k].split(','))
        x_inds = np.array(['x' in x for x in acc_records])
        y_inds = np.array(['y' in x for x in acc_records])
        z_inds = np.array(['z' in x for x in acc_records])
        assert np.sum(x_inds) == np.sum(y_inds),\
            'Number of x and y acceleration data points not the same!'
        assert np.sum(y_inds) == np.sum(z_inds),\
            'Number of y and z acceleration data points not the same!'
        acc_data = np.array([re.sub("[^0-9]", "", x) for x in acc_records], dtype=float)
        x_acc = acc_data[x_inds]
        y_acc = acc_data[y_inds]
        z_acc = acc_data[z_inds]
        acc_mat = np.array([x_acc, y_acc, z_acc]).T
        if not acc_mat.shape[0] == acc_samp[k]:
            warnings.warn('Not correct number of samples in df entry. Presuming they are in the beginning of the'
                          ' measurement period.')
        acc_segment = np.array([[[np.nan] * 3] * acc_samp[k]]).squeeze()
        acc_segment[:len(acc_mat), :] = acc_mat
        acc_signal = np.vstack((acc_signal, acc_segment))
        device = np.append(device, np.array(df['Device ID'][k]).repeat(acc_samp[k]))
        times = np.append(times, np.array(df['Record Time'][k]).repeat(acc_samp[k]) + (np.arange(acc_samp[k]) * 1/acc_samp[k] * 1000))
        sfreq = np.append(sfreq, np.ones(acc_samp[k]) * acc_samp[k])
    triaxial_acc = []
    for c, axis in enumerate(['x', 'y', 'z']):
        df_acc = pd.DataFrame()
        df_acc['Time Unix (ms)'] = times
        df_acc['acc_raw_' + axis] = acc_signal[:, c]
        df_acc['Sampling Frequency (Hz)'] = sfreq
        df_acc['Device ID'] = device
        triaxial_acc.append(df_acc)
    return triaxial_acc


def empatica_parse_acc_data(fdir_raw):
    from avro.datafile import DataFileReader
    from avro.io import DatumReader

    fnames = os.listdir(fdir_raw)
    times_all = np.array([])
    acc_raw_all = np.array([]).reshape((0, 3))
    devices_all = []
    sfreq_all = []
    for fname in fnames:
        if fname[-5:] == '.avro':
            reader = DataFileReader(open(fdir_raw + fname, "rb"), DatumReader())
            data_list = []
            for data_frac in reader:
               data_list.append(data_frac)
            reader.close()
            if not data_list[0]['rawData']['accelerometer']['timestampStart'] == 0: # Avoid empty files
                times = data_list[0]['rawData']['accelerometer']['timestampStart'] / 10 ** 6 + data_list[0]['rawData'][
                    'accelerometer']['samplingFrequency'] ** -1 * np.arange(len(data_list[0]['rawData']['accelerometer']['x']))
                acc_raw = np.array([data_list[0]['rawData']['accelerometer']['x'],
                                    data_list[0]['rawData']['accelerometer']['y'],
                                    data_list[0]['rawData']['accelerometer']['z']]).T
                acc_raw = acc_raw * data_list[0]['rawData']['accelerometer']['imuParams']['physicalMax'] / \
                          data_list[0]['rawData']['accelerometer']['imuParams']['digitalMax']
                devices_all = devices_all + [data_list[0]['deviceSn']] * len(times)
                sfreq_all = sfreq_all + [data_list[0]['rawData']['accelerometer']['samplingFrequency']] * len(times)
                times_all = np.concatenate((times_all, times))
                acc_raw_all = np.concatenate((acc_raw_all, acc_raw), axis=0)
    sort_inds = np.argsort(times_all)
    times_all = times_all[sort_inds]
    acc_raw_all = acc_raw_all[sort_inds]
    devices_all = np.array(devices_all)[sort_inds]
    sfreq_all = np.array(sfreq_all)[sort_inds]

    dfs = []
    for c, axis in enumerate(['x', 'y', 'z']):
        df_acc_raw = pd.DataFrame(data={'acc_raw_' + axis : acc_raw_all[:, c]}, index=times_all)
        df_acc_raw['Device ID'] = devices_all
        df_acc_raw['Sampling_Freq'] = convert_sfreq_to_sampling_interval(sfreq_all)
        df_acc_raw.index = from_unix(df_acc_raw.index, time_unit='s')
        dfs.append(df_acc_raw)

    return dfs