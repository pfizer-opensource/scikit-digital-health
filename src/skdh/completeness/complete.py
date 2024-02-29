import os
import pickle
import numpy as np
from skdh.base import BaseProcess
from skdh.completeness.utils import load_subject_data, compute_completeness_master, compute_summary_metrics, check_hyperparameters
from skdh.completeness.visualizations import visualize_overview_plot, plot_completeness, plot_data_gaps, plot_timescale_completeness


class completeness_pipe(BaseProcess):
    r"""
    Pipeline for assessing signal completeness.
    """

    def __init__(
            self,
            subject,
            subject_folder,
            device_name,
            fpath_output,
            measures,
            resample_width_mins=5,
            gap_size_mins=30,
            ranges=None,
            data_gaps=None,
            time_periods=None,
            timescales=None):

        check_hyperparameters(subject,
            subject_folder,
            device_name,
            fpath_output,
            measures,
            resample_width_mins,
            gap_size_mins,
            ranges,
            data_gaps,
            time_periods,
            timescales)

        super().__init__(
            subject=subject,
            subject_folder=subject_folder,
            device_name=device_name,
            fpath_output=fpath_output,
            measures=measures,
            resample_width=resample_width_mins,
            gap_size_mins=gap_size_mins,
            ranges=ranges,
            data_gaps=data_gaps,
            time_periods=time_periods,
            timescales=timescales)

        self.subject = subject
        self.subject_folder = subject_folder
        self.device_name = device_name
        self.fpath_output = fpath_output
        self.measures = measures
        self.resample_width_mins = resample_width_mins
        self.gap_size_mins = gap_size_mins
        self.ranges = ranges
        self.data_gaps = data_gaps
        self.time_periods = time_periods
        self.timescales = timescales

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
        data_dic = load_subject_data(self.subject_folder, self.subject, self.device_name, self.measures, self.ranges)

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
                                                   resample_width_mins=self.resample_width_mins, gap_size_mins=self.gap_size_mins,
                                                   time_periods=self.time_periods)
            reason_color_dic = {'normal' : 'blue', 'unknown' : 'magenta', 'charging' : 'orange', 'no_wear' : 'red'}
            figures.update({'overview': overview_fig})
            completeness_fig = plot_completeness(completeness_master_dic, data_dic, self.time_periods,
                                                 fpath=fpath_dir + 'completeness', reason_color_dic=reason_color_dic)
            figures.update({'Completeness': completeness_fig})
            if not self.data_gaps is None:
                data_gap_fig = plot_data_gaps(completeness_master_dic, data_dic, self.data_gaps, self.time_periods,
                                              fpath=fpath_dir + 'data_gaps', reason_color_dic=reason_color_dic)
                figures.update({'data_gaps': data_gap_fig})
            if not self.timescales is None:
                timescale_compl_fig = plot_timescale_completeness(completeness_master_dic, data_dic, self.time_periods,
                                                                  self.timescales,
                                                                  fpath=fpath_dir + 'timescale_completeness',
                                                                  reason_color_dic=reason_color_dic)
                figures.update({'timescale_completeness': timescale_compl_fig})

        return {"Completeness": {'data_dic' : data_dic, 'compl_dic' : completeness_master_dic,
                                 'df_summary' : df_summary, 'figures' : figures}}

