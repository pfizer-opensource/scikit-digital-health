import copy
import warnings

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import skdh

def plot_overview_one_device(data, resample_width_mins=1, device_changes=None, shared_xaxes=True, title=None,
                             gap_size_mins=5):
    r"""Version of plot overview that plots several data streams for only one device/subject.

    :param data: list where each element is a df with one column and a time stamp index. Each df will be plotted in one
    row and column title will be plotted as y-title.
    """
    y_titles = [ele.name for ele in data]
    fig = make_subplots(rows=len(data), cols=1, x_title='Date', shared_xaxes=shared_xaxes)
    colors = px.colors.qualitative.Alphabet

    for c, data_section in enumerate(data):
        first_df = data_section.resample(str(resample_width_mins) + 'min').median()
        lo, so, vo = skdh.utility.internal.rle(np.isnan(np.array(first_df)))
        start_nan_inds = so[vo]
        end_nan_inds = so[vo] + lo[vo] - 1
        nan_region_lengths = lo[vo]

        scatter_points = end_nan_inds[np.where((start_nan_inds[1:] - end_nan_inds[:-1]) == 2)[0]] + 1
        crit_data_gap_inds = np.where((nan_region_lengths + 1) * resample_width_mins >= gap_size_mins)[0]
        data_gaps_start_ind = start_nan_inds[crit_data_gap_inds]
        data_gaps_end_ind = end_nan_inds[crit_data_gap_inds]
        if len(data_gaps_start_ind) > 100:
            warnings.warn('The chosen data gap length (' + str(gap_size_mins) + ' min) to highlight resulted in >100'
                                                                                'data gaps. This will take some time to'
                                                                                ' render, consider setting gap_size_'
                                                                                'mins higher to make it faster')

        fig.add_trace(go.Scatter(x=first_df.index, y=first_df, line=dict(color=colors[c])), row=c + 1, col=1)
        if len(scatter_points) > 0:
            fig.add_trace(go.Scatter(x=first_df.index[scatter_points], y=first_df[scatter_points], mode='markers',
                                     marker={'symbol' : 'line-ew-open', 'color' : colors[c], 'size' : 2}), row=c + 1, col=1)
        for start, end in zip(data_gaps_start_ind, data_gaps_end_ind):
            if end == len(first_df) - 1: end = end - 1 # edge case of data gap going to the end of the signal
            if start == 0: start = start + 1 # edge case of data gap starting in beginning of signal
            fig.add_vrect(x0=first_df.index[start - 1], x1=first_df.index[end + 1],
                          fillcolor='red', opacity=0.25, line_width=0, row=c + 1, col=1)
        if len(data_gaps_start_ind) > 0:
            fig.add_trace(go.Scatter(x=[first_df.index[data_gaps_start_ind[0]], first_df.index[data_gaps_end_ind[0]]],
                                     y=[first_df.index[0], first_df.index[0]], line=dict(color='red'),
                                     name='Data gap > ' + str(gap_size_mins) + ' min',
                                     mode='lines', opacity=0.25))
        if device_changes is not None:
            for dev_change in device_changes[c][0]:
                try:
                    fig.add_vline(x=data_section[0].index[dev_change + 1], line_color='green', row=c + 1, col=1)
                    fig.add_vline(x=data_section[0].index[dev_change], line_color='red', row=c + 1, col=1)
                    fig.add_trace(go.Scatter(x=[data_section[0].index[dev_change], data_section[0].index[dev_change]],
                                             y=[data_section[0].index[0], data_section[0].index[0]], line=dict(color='green'),
                                             name='Beginning of device use for ' + y_titles[0], mode='lines'), row=c + 1, col=1)
                    fig.add_trace(go.Scatter(x=[data_section[0].index[dev_change], data_section[0].index[dev_change]],
                                             y=[data_section[0].index[0], data_section[0].index[0]], line=dict(color='red'),
                                             name='End of device use for ' + y_titles[0], mode='lines'), row=c + 1, col=1)
                except IndexError: 'no device changes in first data stream'
        fig['layout']['yaxis' + str((c + 1))]['title'] = y_titles[c]

    if not title is None:
        fig.update_layout(title=title)
    remove_duplicate_labels_plotly(fig)
    return fig


def remove_duplicate_labels_plotly(fig):
    names = set()
    fig.for_each_trace(
        lambda trace:
        trace.update(showlegend=False)
        if (trace.name in names) else names.add(trace.name))

    return fig


def plot_timescale_completeness(completeness_master_dic, data_dic, time_periods, timescales, fpath, figsize=(12, 12), dpi=100, reason_color_dic=None):
    timescales = ['native'] + [x for x in timescales]
    if reason_color_dic is None:
        reason_color_dic = {}
        prior_reason_dic = False
    else: prior_reason_dic = True
    fig = plt.figure(figsize=figsize, dpi=dpi)
    axes = fig.subplots(len(data_dic['Measurement Streams'].keys()), len(time_periods), sharex=True)
    x_labels = [str(timescale) for timescale in timescales]
    handles = []
    labels = []
    if len(time_periods) == 1 or len(data_dic['Measurement Streams'].keys()) == 1:
        axes = np.array([axes]).reshape((len(data_dic['Measurement Streams']), len(time_periods)))
    for c, time_period in enumerate(time_periods):
        for d, measure in enumerate(list(data_dic['Measurement Streams'].keys())):
            heights = [completeness_master_dic[measure]['Completeness'][timescale][time_period]['Completeness'] * 100
                       for timescale in timescales]
            axes[d, c].bar(x=np.arange(len(timescales)) * 3., height=heights, color='black', label='Completeness', zorder=2)
            if d == len(data_dic['Measurement Streams'].keys()) - 1:
                axes[d, c].set_xticks(ticks=np.arange(len(timescales)) * 3 + .5, labels=x_labels, rotation=45)
            else:
                axes[d, c].set_xticks(ticks=np.arange(len(timescales)) * 3 + .5, labels=[])
            ax2 = axes[d, c].twinx()
            ax2.bar(x=0, height=0, color='black', label='Completeness', zorder=2)
            for k, timescale in enumerate(timescales):
                bottom = 0
                for reason, quota in completeness_master_dic[measure]['Completeness'][timescale][time_period].items():
                    if not reason == 'Completeness':
                        if not prior_reason_dic: reason_color_dic = grow_reason_color_dict(reason.replace('Missingness, ', ''), reason_color_dic)
                        ax2.bar(x=k * 3. + 1, height=quota * 100, width=.8, bottom=bottom, linewidth=1,
                                edgecolor='black', color=reason_color_dic[reason.replace('Missingness, ', '')], zorder=1,
                                label=reason.replace(' ', '\n') + ' reason')
                        bottom = bottom + quota * 100
            handles = handles + fig.gca().get_legend_handles_labels()[0]
            labels = labels + fig.gca().get_legend_handles_labels()[1]
            axes[d, c].set_ylim((0, 100))
            ax2.set_ylim((0, 100))
            axes[d, c].grid(axis='y', linestyle='--', color='gray', zorder=1000)
            if d == 0:
                axes[d, c].set_title(str(time_period[0]) + ' - \n' + str(time_period[1]))
            if d == axes.shape[0] - 1:
                axes[d, c].set_xlabel('Timescale')
            if c == 0:
                axes[d, c].set_ylabel(list(data_dic['Measurement Streams'].keys())[d] + '\nCompleteness\n(%)')
                if d == 0:
                    first_ax2 = ax2
    by_label = dict(zip(labels, handles))
    first_ax2.legend(by_label.values(), by_label.keys(), loc='upper left')
    fig.text(0.985, 0.55, 'Missingness (%) by reason', va='center', rotation=270)
    fig.tight_layout()
    fig.savefig(fpath + '.png', dpi=300)
    fig.savefig(fpath + '.svg')

    return fig


def plot_completeness(completeness_master_dic, data_dic, time_periods, fpath, figsize=None, dpi=100, reason_color_dic=None):
    if reason_color_dic is None:
        reason_color_dic = {}
        prior_reason_dic = False
    else: prior_reason_dic = True
    if figsize is None:
        figsize = (6 * len(data_dic['Measurement Streams'].keys()), 8)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    axes = fig.subplots(1, len(data_dic['Measurement Streams'].keys()))
    if len(data_dic['Measurement Streams'].keys()) == 1:
        axes = np.array([axes])
    x_labels = [str(time_period[0]) + '-\n' + str(time_period[1]) for time_period in time_periods]
    handles = []
    labels = []
    for d, measure in enumerate(list(data_dic['Measurement Streams'].keys())):
        ax2 = axes[d].twinx()
        ax2.bar(x=0, height=0, color='black', label='Completeness', zorder=2)
        for c, time_period in enumerate(time_periods):
            axes[d].bar(x=c * 3., width=.8, height=completeness_master_dic[measure]['Completeness']['native'][time_period][
                                                  'Completeness'] * 100, color='black', label='Completeness', zorder=2)
            bottom = 0
            for reason, quota in completeness_master_dic[measure]['Completeness']['native'][time_period].items():
                if not reason == 'Completeness':
                    if not prior_reason_dic: reason_color_dic = grow_reason_color_dict(reason, reason_color_dic)
                    ax2.bar(x=c * 3. + 1, height=quota * 100, width=.8, bottom=bottom, linewidth=1,
                            edgecolor='black', color=reason_color_dic[reason.replace('Missingness, ', '')], zorder=1,
                            label=reason.replace(' ', '\n') + ' reason')
                    bottom = bottom + quota * 100
        axes[d].set_title(measure)
        axes[d].set_xlabel('Time period' )
        if d == 0:
            first_ax2 = ax2
            axes[d].set_ylabel('Completeness (%)')
        else:
            axes[d].set_yticks(ticks=np.arange(6) * 20, labels=[])
        handles = handles + fig.gca().get_legend_handles_labels()[0]
        labels = labels + fig.gca().get_legend_handles_labels()[1]
        axes[d].set_xticks(ticks=np.arange(len(time_periods)) * 3 + .5, labels=x_labels, rotation=90)
        axes[d].set_ylim((0, 100))
        ax2.set_ylim((0, 100))
        if d == len(data_dic['Measurement Streams'].keys()) - 1:
            ax2.set_ylabel('Missingness by reason (%)', rotation=270)
        else:
            ax2.set_yticks(ticks=np.arange(6) * 20, labels=[])
        axes[d].grid(axis='y', linestyle='--', color='gray', zorder=100)
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    first_ax2.legend(by_label.values(), by_label.keys(), loc='upper left')
    fig.tight_layout()
    fig.savefig(fpath + '.png', dpi=300)
    fig.savefig(fpath + '.svg')

    return fig

def grow_reason_color_dict(reason, reason_color_dic):
    if reason not in reason_color_dic.keys():
        reason_color_dic.update({reason: list(plt.cm.get_cmap('tab20').colors)[len(reason_color_dic.keys())]})
    return reason_color_dic

def plot_data_gaps(completeness_master_dic, data_dic, data_gaps, time_periods, fpath, figsize=(12, 12), dpi=100, reason_color_dic=None):
    if reason_color_dic is None:
        reason_color_dic = {}
        prior_reason_dic = False
    else: prior_reason_dic = True
    fig = plt.figure(figsize=figsize, dpi=dpi)
    axes = fig.subplots(len(data_dic['Measurement Streams'].keys()), len(time_periods))
    x_labels = [str(dg) for dg in data_gaps]
    handles = []
    labels = []
    if len(time_periods) == 1 or len(data_dic['Measurement Streams'].keys()) == 1:
        axes = np.array([axes]).reshape((len(data_dic['Measurement Streams']), len(time_periods)))
    for c, time_period in enumerate(time_periods):
        for d, measure in enumerate(list(data_dic['Measurement Streams'].keys())):
            counts_total = [np.sum(list(list(completeness_master_dic[measure]['data_gaps'][time_period].values())[k].values())) for k in range(len(data_gaps))]
            axes[d, c].bar(x=np.arange(len(data_gaps)) * 3., height=counts_total, color='black', zorder=2)
            if d == len(data_dic['Measurement Streams'].keys()) - 1:
                axes[d, c].set_xticks(ticks=np.arange(len(data_gaps)) * 3 + .5, labels=x_labels, rotation=45)
            else:
                axes[d, c].set_xticks(ticks=np.arange(len(data_gaps)) * 3 + .5, labels=[])
            ax2 = axes[d, c].twinx()
            ax2.bar(x=0, height=0, color='black', label='Data gaps (n)', zorder=2)
            for k, data_gap in enumerate(completeness_master_dic[measure]['data_gaps'][time_period].values()):
                bottom = 0
                for reason, quota in data_gap.items():
                    reason_ratio = quota / counts_total[k] * 100
                    if not prior_reason_dic: reason_color_dic = grow_reason_color_dict(reason, reason_color_dic)
                    ax2.bar(x = k * 3 + 1, height=reason_ratio, width=.8, log=False,
                            color=reason_color_dic[reason.replace('Missingness, ', '')], bottom=bottom,
                            linewidth=1, edgecolor='black', zorder=1, label='Gap reason (%):\n' + reason)
                    bottom = bottom + reason_ratio
            if d == 0:
                axes[d, c].set_title(str(time_period[0]) + ' - \n' + str(time_period[1]))
            if d == axes.shape[0] - 1:
                axes[d, c].set_xlabel('Data gap duration' )
            if c == 0:
                axes[d, c].set_ylabel(list(data_dic['Measurement Streams'].keys())[d] + '\nData gaps per\ntime period (N)')
                if d == 0:
                    first_ax2 = ax2
            axes[d, c].set_xlim((-.5, (len(data_gaps) - 1) * 3 + 1.5))
            ax2.set_ylim((0, 100))
            handles = handles + fig.gca().get_legend_handles_labels()[0]
            labels = labels + fig.gca().get_legend_handles_labels()[1]
    by_label = dict(zip(labels, handles))
    first_ax2.legend(by_label.values(), by_label.keys(), loc='upper left')
    fig.text(0.985, 0.55, 'Gap reason (%)', va='center', rotation=270)
    fig.tight_layout()
    fig.savefig(fpath + '.png', dpi=300)
    fig.savefig(fpath + '.svg')
    return fig


def visualize_overview_plot(data_dic, fpath, resample_width_mins, gap_size_mins, time_periods, fontsize=None):
    data_streams = [copy.deepcopy(data_dic['Measurement Streams'][key][key]) for key in data_dic['Measurement Streams'].keys()]
    for x in ['Wear Indicator', 'Charging Indicator']:
        if x in data_dic.keys():
            data_streams.append(copy.deepcopy(data_dic[x][x]))
    acc_raw_ind = np.where([data_stream.name == 'acc_raw' for data_stream in data_streams])[0]
    if len(acc_raw_ind) > 0:
        data_streams[acc_raw_ind[0]].iloc[:] = np.linalg.norm(np.array([np.array(x, dtype=float) for x in data_streams[acc_raw_ind[0]]]), axis=1)
    fig = plot_overview_one_device(data_streams, resample_width_mins=resample_width_mins, device_changes=None, shared_xaxes=True,
                                   title=data_dic['Subject ID'], gap_size_mins=gap_size_mins)
    for row_ind, measure in enumerate(data_dic['Measurement Streams'].keys()):
        for time_period in time_periods:
            fig.add_vline(x=time_period[0], line_color='green', row=row_ind + 1, col=1)
            fig.add_vline(x=time_period[1], line_color='red', row=row_ind + 1, col=1)
            fig.add_trace(go.Scatter(x=[time_period[0], time_period[0]], y=[time_period[0], time_period[0]],
                                     line=dict(color='green'), name='Beginning of time period', mode='lines'),
                          row=row_ind + 1, col=1)
            fig.add_trace(go.Scatter(x=[time_period[0], time_period[0]], y=[time_period[0], time_period[0]],
                                     line=dict(color='red'), name='End of time period', mode='lines'),
                          row=row_ind + 1, col=1)
    remove_duplicate_labels_plotly(fig)
    if not fontsize is None:
        fig.update_yaxes(title_font=dict(size=fontsize))
    fig.write_html(fpath)
    return fig
