import os
import pickle
import numpy as np
from skdh.base import BaseProcess
from skdh.completeness.parse import parse_raw
from skdh.completeness.utils import input_data_checks, init_data_dic, compute_completeness_master, compute_summary_metrics
from skdh.completeness.visualizations import visualize_overview_plot, plot_completeness, plot_data_gaps, plot_timescale_completeness


class completeness_pipe(BaseProcess):
    r"""
    Pipeline for assessing signal completeness.
    """

    def __init__(
            self,
            subject_folder,
            device_name,
            fpath_output,
            unix_time_key,
            columns,
            measures,
            subject_id_key,
            resample_width='5m',
            gap_size_mins=30,
            ranges={},
            timezone_key=None,
            data_gaps=None,
            time_periods=None,
            timescales=None,
            ecg_key=None,
            acc_raw_key=None,
            acc_raw_fdir=None):

        self.df_raw = parse_raw(subject_folder, unix_time_key, subject_id_key, timezone_key=timezone_key)
        input_data_checks(self.df_raw, device_name)

        super().__init__(
            subject_folder=subject_folder,
            device_name=device_name,
            fpath_output=fpath_output,
            unix_time_key=unix_time_key,
            columns=columns,
            measures=measures,
            subject_id_key=subject_id_key,
            resample_width=resample_width,
            gap_size_mins=gap_size_mins,
            ranges=ranges,
            timezone_key=timezone_key,
            data_gaps=data_gaps,
            time_periods=time_periods,
            timescales=timescales,
            ecg_key=ecg_key,
            acc_raw_key=acc_raw_key,
            acc_raw_fdir=acc_raw_fdir)

        self.subject_folder = subject_folder
        self.device_name = device_name
        self.fpath_output = fpath_output
        self.unix_time_key = unix_time_key
        self.columns = columns
        self.measures = measures
        self.subject_id_key = subject_id_key
        self.resample_width = resample_width
        self.gap_size_mins = gap_size_mins
        self.ranges = ranges
        self.timezone_key = timezone_key
        self.data_gaps = data_gaps
        self.time_periods = time_periods
        self.timescales = timescales
        self.ecg_key = ecg_key
        self.acc_raw_key = acc_raw_key
        self.acc_raw_fdir = acc_raw_fdir

#    @handle_process_returns(results_to_kwargs=True)
    def predict(self,
                generate_figures=True,
                **kwargs):
        """
        Compute completeness and save results. Create and save figures if generate_figures is True.
        """
        super().predict(
            expect_days=False,
            expect_wear=False,
            **kwargs,
        )

        data_dic = init_data_dic(self.df_raw, self.columns, self.measures, self.device_name, self.ranges, self.ecg_key,
                                 self.acc_raw_key, self.acc_raw_fdir)
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
        completeness_master_dic = compute_completeness_master(data_dic, data_gaps=self.data_gaps, time_periods=self.time_periods,
                                                              timescales=self.timescales)

        # Save raw results and summary metrics
        os.system('mkdir ' + self.fpath_output + '/' + data_dic['Subject ID'])
        os.system('mkdir ' + self.fpath_output + '/' + data_dic['Subject ID'] + '/' + self.device_name)
        pickle.dump(completeness_master_dic,
                    open(self.fpath_output + '/' + data_dic['Subject ID'] + '/' + self.device_name + '/raw_completeness', 'wb'))
        df_summary = compute_summary_metrics(completeness_master_dic, self.time_periods, self.timescales,
                                             self.measures)  # Daily wear time, charging, data gaps, native completeness
        df_summary.to_csv(self.fpath_output + '/' + data_dic['Subject ID'] + '/' + self.device_name + '/summary_metrics.csv')

        # Create and save visualizations of results
        figures = {}
        if generate_figures:
            fpath_dir = self.fpath_output + '/' + data_dic['Subject ID'] + '/' + self.device_name + '/'
            overview_fig = visualize_overview_plot(data_dic=data_dic, fpath=fpath_dir + 'overview.html',
                                                   resample_width=self.resample_width, gap_size_mins=self.gap_size_mins,
                                                   time_periods=self.time_periods)
            figures.update({'overview': overview_fig})
            completeness_fig = plot_completeness(completeness_master_dic, data_dic, self.time_periods,
                                                 fpath=fpath_dir + 'completeness', reason_color_dic=None)
            figures.update({'completeness': completeness_fig})
            if not self.data_gaps is None:
                data_gap_fig = plot_data_gaps(completeness_master_dic, data_dic, self.data_gaps, self.time_periods,
                                              fpath=fpath_dir + 'data_gaps', reason_color_dic=None)
                figures.update({'data_gaps': data_gap_fig})
            if not self.timescales is None:
                timescale_compl_fig = plot_timescale_completeness(completeness_master_dic, data_dic, self.time_periods,
                                                                  self.timescales,
                                                                  fpath=fpath_dir + 'timescale_completeness',
                                                                  reason_color_dic=None)
                figures.update({'timescale_completeness': timescale_compl_fig})

        return {"completeness": {'df_parsed' : self.df_raw, 'data_dic' : data_dic, 'compl_dic' : completeness_master_dic,
                                 'df_summary' : df_summary, 'figures' : figures}}

