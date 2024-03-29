import numpy as np


# Test input check on real data
def test_1_load_data(data_dic):

    assert list(data_dic.keys()) == ['Subject ID', 'Measurement Streams', 'Wear Indicator'] and \
           'Time Unix (ms)' in list(data_dic['Wear Indicator'].keys()) and \
           'Wear Indicator' in list(data_dic['Wear Indicator'].keys()) and \
           'Sampling Frequency (Hz)' in list(data_dic['Wear Indicator'].keys()) and \
           'Device ID' in list(data_dic['Wear Indicator'].keys()) and \
           'Heart Rate' in list(data_dic['Measurement Streams'].keys()) and \
           'Resp Rate' in list(data_dic['Measurement Streams'].keys())

def test_2_wear_data(pipe, data_dic):
    measures = ['Resp Rate', 'Heart Rate']
    completeness = pipe.predict(data_dic=data_dic, measures=measures, fpath_output=None)

    assert completeness['Completeness']['df_summary'][
               ('2024-03-01 06:56:41.588999936', '2024-03-01 12:54:49.588999936')]['wear time'] == np.timedelta64(
        19268000000000,'ns') and \
           completeness['Completeness']['df_summary'][
               ('2024-03-01 06:56:41.588999936', '2024-03-01 12:54:49.588999936')]['no-wear time'] == np.timedelta64(
        1980000000, 'us')

def test_3_resp_rate(pipe, data_dic):
    measures = ['Resp Rate', 'Heart Rate']
    completeness = pipe.predict(data_dic=data_dic, measures=measures, fpath_output=None)

    assert completeness['Completeness']['df_summary'][
               ('2024-03-01 06:56:41.588999936', '2024-03-01 12:54:49.588999936')][
        'Resp Rate completeness: 0 (no valid values)'] == 0

def test_4_heart_rate_completeness(pipe, data_dic):
    measures = ['Resp Rate', 'Heart Rate']
    completeness = pipe.predict(data_dic=data_dic, measures=measures, fpath_output=None)

    assert completeness['Completeness']['df_summary'][
               ('2024-03-01 06:56:41.588999936', '2024-03-01 12:54:49.588999936')][
        'Heart Rate, Completeness, native'] == 0.8295175446790767 and \
           completeness['Completeness']['df_summary'][
               ('2024-03-01 06:56:41.588999936', '2024-03-01 12:54:49.588999936')][
               'Heart Rate, Missingness, no_wear, native'] == 0.0765921967398548 and \
            completeness['Completeness']['df_summary'][
                ('2024-03-01 06:56:41.588999936', '2024-03-01 12:54:49.588999936')][
                'Heart Rate, Completeness, 1 minutes'] == 0.5835734365722264

def test_5_heart_rate_datagaps(pipe, data_dic):
    measures = ['Resp Rate', 'Heart Rate']
    completeness = pipe.predict(data_dic=data_dic, measures=measures, fpath_output=None)

    assert completeness['Completeness']['df_summary'][
        ('2024-03-01 06:56:41.588999936', '2024-03-01 12:54:49.588999936')][
        'Heart Rate, data gap 10 minutes, reason: unknown'] == 3 and \
        completeness['Completeness']['df_summary'][
                ('2024-03-01 06:56:41.588999936', '2024-03-01 12:54:49.588999936')][
                'Heart Rate, data gap 10 minutes, reason: no_wear'] == 1 and \
        completeness['Completeness']['df_summary'][
                ('2024-03-01 06:56:41.588999936', '2024-03-01 12:54:49.588999936')][
                'Heart Rate, data gap 30 minutes, reason: unknown'] == 0 and \
        completeness['Completeness']['df_summary'][
                ('2024-03-01 06:56:41.588999936', '2024-03-01 12:54:49.588999936')][
                'Heart Rate, data gap 30 minutes, reason: no_wear'] == 1
