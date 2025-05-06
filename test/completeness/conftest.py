from pathlib import Path

from pytest import fixture
from pandas import read_csv
import numpy as np
import skdh

@fixture(scope="module")
def completeness_sub_data():
    cwd = str(Path.cwd())
    if cwd.split('/')[-1] == "completeness":
        subject_folder = cwd + '/data/'
    elif cwd.split('/')[-1] == "test":
        subject_folder = cwd + '/completeness/data/'
    elif cwd.split('/')[-1] == "scikit-digital-health":
        subject_folder = cwd + '/test/completeness/data/'

    measures = ['Resp Rate', 'Heart Rate']
    ranges = {'Resp Rate': [0, 100], 'Heart Rate': [20, 230]}
    timescales = np.array([np.timedelta64(1, 'm'), np.timedelta64(30, 'm'), np.timedelta64(1, 'h')])
    data_gaps = np.array([np.timedelta64(10, 'm'), np.timedelta64(30, 'm'), np.timedelta64(1, 'h')])
    subject = 'test_sub'
    time_periods = 'daily'
    pipe = skdh.completeness.AssessCompleteness(ranges, data_gaps, time_periods, timescales)
    data_dic = pipe.load_subject_data(subject_folder, subject, measures)

    return pipe, data_dic

