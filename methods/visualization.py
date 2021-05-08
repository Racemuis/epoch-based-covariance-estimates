# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 14:42:02 2021

@author: Jan Sosulski
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.serif'] = ['Computer Modern Roman']

path = r'./spot_single_subj_all_sess_all_full_shrinkage_results.csv'
dataframe = pd.read_csv(path)

def plot_benchmark_results(results, dataset=None, dim_prefix='', jm_dim=None, ylim=None, save=False, output_dir=None,
                           out_prefix=None, session_mean=False, figsize=(6, 4), plot_legend=False):
    pipes = results['pipeline'].unique()
    custom_order = ['Tangent space LDA shrinkage', 'Response shrinkage', 'Prototype shrinkage']
    if session_mean:
        results = results.groupby(['subject', 'pipeline']).aggregate([np.mean]).reset_index()
        results.columns = [x[0] for x in results.columns]
    ax, leg_handles,mean_handle = plot_matched_wrapper(data=results, x='pipeline', y='score', x_order = custom_order, match_col = 'subject',
                                           figsize=figsize, x_match_sort = 'Response shrinkage')
    if plot_legend:
        leg_labels = [str(i) for i in range(1, len(leg_handles) + 1)]
        first_legend = plt.legend(handles=mean_handle, labels = ['mean'],title = 'mean', loc='lower left', bbox_to_anchor=(0.84, -.16), prop={'size': 7})

        # Add the legend manually to the current Axes.
        plt.gca().add_artist(first_legend)
        # Create another legend
        plt.legend(handles=leg_handles, labels=leg_labels, title='Subj.', ncol=1, prop={'size': 8},
                   loc='center left', bbox_to_anchor=(1, 0.5))
        
    ax.set_ylabel('ROC AUC')
    ax.set_xlabel('Classification method')
    title = r'SPOT dataset'
    ax.set_title(title)
    fig = ax.figure
    fig.tight_layout()
    plt.grid(axis = 'both')
    if type(ylim) is tuple and len(ylim) == 2:
        ax.set_ylim(ylim[0], ylim[1])
    elif ylim == 'auto':
        ax.set_ylim(auto=True)

def plot_matched_wrapper(data=None, x=None, y=None, x_order=None, match_col=None, x_match_sort=None, title=None,
                         ax=None, figsize=(9, 6)):
    if ax is None:
        fig, ax = plt.subplots(1, 1, facecolor='white', figsize=figsize, )
    if match_col is not None:
        ax, leg_handles, mean_handle = plot_matched(data=data, x=x, y=y, x_order=x_order, match_col=match_col,
                                       x_match_sort=x_match_sort, title=title, ax=ax, figsize=figsize, sort_marker='')
    else:
        grouped = data.groupby(['pipeline', 'subject'], as_index = False).mean()
        ax = sns.stripplot(data=grouped, y=y, x=x, ax=ax, jitter=True, alpha=0.2, zorder=1,
                           order=x_order)
        leg_handles = None
    return ax, leg_handles, mean_handle

def plot_matched(data=None, x=None, y=None, x_order=None, match_col=None, x_match_sort=None, title=None, ax=None,
                 figsize=(9, 6), sort_marker='●', error='amend'):
    if ax is None:
        fig, ax = plt.subplots(1, 1, facecolor='white', figsize=figsize)
    cp = sns.color_palette()
    marker_arr = ['^', 's', 'p', 'o']
    n_markers = len(marker_arr)
    num_x = len(data[x].unique())
    num_matched = len(data[match_col].unique())
    sort_idx = None
    if x_match_sort is not None:
        sort_idx = data.loc[data[x] == x_match_sort].sort_values(by=match_col, ascending=True).reset_index().\
            sort_values(by='score', ascending=True).index.copy()
    c_offs_left = 2
    c_offs_right = 2
    gmap = LinearSegmentedColormap.from_list('custom', [(0, 0, 0), (0.5, 0.5, 0.5), (1, 1, 1)],
                                             N=(num_matched // n_markers + 1 + c_offs_left + c_offs_right))
    legend_markers = []
    for i, x_main in enumerate(x_order):
        base_col = cp[i]
        cmap = LinearSegmentedColormap.from_list('custom', [(0, 0, 0), base_col, (1, 1, 1)],
                                                 N=(num_matched // n_markers + 1 + c_offs_left + c_offs_right))
        r = data.loc[data[x] == x_main] 
        print(r)
        if sort_idx is not None:
            r = r.iloc[sort_idx]
        x_center = i + 1
        x_width = 0.20
        x_space = np.linspace(x_center - x_width, x_center + x_width, num_matched)
        m_score, m_std = r.aggregate((np.mean, np.std))[y]
        m_err = m_std / np.sqrt(num_matched)
        print("The average ROC AUC for {} is {}, with a SEM of {}.\n".format(i, m_score, m_err))
        err_artists = ax.errorbar(x_center, m_score, m_err, capsize=0, zorder=15, color='k', linewidth=1, alpha=1,
                                  dash_capstyle='round')
        err_artists[2][0].set_capstyle('round')
        ax.scatter(x_center, m_score, marker='X', color=base_col, edgecolor='k', linewidth=0.4, zorder=20, s=50)
        for j, x_j in enumerate(x_space):
            score = r[y].iloc[j]
            m = marker_arr[j % len(marker_arr)]
            sdef = 70
            s = sdef * 0.66 if m in ['s', 'D'] else sdef  # these two markers are unexplicably larger in default pyplot
            ax.scatter(x_j, score, alpha=.8, marker=m, linewidth=0.4, edgecolor=(0.8, 0.8, 0.8), s=s,
                       color=cmap((j // n_markers) + c_offs_left))
            if i == 0:
                legend_markers.append(Line2D([0], [0], marker=m, color='w', label=r[match_col].iloc[j],
                                      markerfacecolor=gmap((j // n_markers) + c_offs_left), markersize=1.2*np.sqrt(s)))
                # need to translate markersize between scatter and plt function, 1.2*sqrt() seems to work kind of
    ax.set_xticks(np.arange(1, num_x + 1))
    if x_match_sort is not None:
        x_order[x_order.index(x_match_sort)] = sort_marker + x_order[x_order.index(x_match_sort)]
    ax.set_xticklabels(x_order)
    ax.set_xlim(0, num_x + 1)
    for i, tick in enumerate(ax.get_xticklabels()):
        if len(tick.get_text()) > 0:
            tick.set_ha('right')
            tick.set_rotation_mode('anchor')
            tick.set_rotation(10)
            tick.set_color(cp[i])
    if title is not None:
        ax.set_title(title)
    return ax, legend_markers, [Line2D([0], [0], marker='X', color='black', markersize=6, linestyle='None')]

plot_benchmark_results(dataframe, plot_legend = True, session_mean = True)
