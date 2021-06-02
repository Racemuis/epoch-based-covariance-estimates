# -*- coding: utf-8 -*-
"""
Copyright 2020 Jan Sosulski

Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the "Software"), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 of the Software, and to permit persons to whom the Software is furnished to do
 so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
 SOFTWARE.

This file has been adapted to sort the legend.
"""

import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


def plot_matched(
    data=None,
    x=None,
    y=None,
    x_order=None,
    match_col=None,
    x_match_sort=None,
    title=None,
    x_xoffset=0.2,
    ax=None,
    figsize=(9, 6),
    sort_marker="⇔",
    error="amend",
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, facecolor="white", figsize=figsize)
    if x_order is not None:
        data = data.loc[data[x].str.match("|".join([re.escape(xo) for xo in x_order]))]
        ux = data[x].unique()
        clean_order = [x_order[i] for i in range(len(x_order)) if x_order[i] in ux]
        if len(clean_order) < len(x_order):
            print("Warning: Truncating ordering, as some pipelines not existing in data frame.")
            x_order = clean_order
        if x_match_sort not in x_order:
            errorwarn_string = (
                f"Cannot sort by {x_match_sort} as it is not in {x_order} / data array."
            )
            if error == "raise":
                raise ValueError(errorwarn_string)
            else:
                print(f"WARNING: {errorwarn_string}")
                x_match_sort = None
    else:
        x_order = list(data[x].unique())
        if x_match_sort is not None:
            x_order.remove(x_match_sort)
            x_order.insert(0, x_match_sort)
    cp = sns.color_palette()
    marker_arr = ["^", "s", "p", "H", "o", "v"]
    n_markers = len(marker_arr)
    num_x = len(data[x].unique())
    num_matched = len(data[match_col].unique())
    sort_idx = None
    legend_labels = data[match_col].unique()
    if x_match_sort is not None:
        sort_idx = (
            data.loc[data[x] == x_match_sort]
            .sort_values(by=match_col, ascending=True)
            .reset_index()
            .sort_values(by=y, ascending=True)
            .index.copy()
        )
        #legend_labels = legend_labels[sort_idx]
    c_offs_left = 2
    c_offs_right = 2
    gmap = LinearSegmentedColormap.from_list(
        "custom",
        [(0, 0, 0), (0.5, 0.5, 0.5), (1, 1, 1)],
        N=(num_matched // n_markers + 1 + c_offs_left + c_offs_right),
    )
    legend_handles = []
    for i, x_main in enumerate(x_order):
        base_col = cp[i]
        cmap = LinearSegmentedColormap.from_list(
            "custom",
            [(0, 0, 0), base_col, (1, 1, 1)],
            N=(num_matched // n_markers + 1 + c_offs_left + c_offs_right),
        )
        r = data.loc[data[x] == x_main].sort_values(by=match_col, ascending=True).reset_index()
        if sort_idx is None:
            sort_idx = r.sort_values(by=match_col, ascending=True).index.copy()
        if sort_idx is not None:
            r = r.iloc[sort_idx]
        legend_labels = r[match_col] #added here
        x_center = i + 1
        x_width = 0.4 - x_xoffset / 2
        x_space = np.linspace(x_center - x_width, x_center + x_width, num_matched)
        m_score, m_std = r.aggregate((np.mean, np.std))[y]
        m_err = m_std / np.sqrt(num_matched)
        err_artists = ax.errorbar(
            x_center,
            m_score,
            2 * m_err,
            capsize=0,
            zorder=15,
            color="k",
            linewidth=1,
            alpha=1,
            dash_capstyle="round",
        )
        err_artists[2][0].set_capstyle("round")
        ax.scatter(
            x_center,
            m_score,
            marker="X",
            color=base_col,
            edgecolor="k",
            linewidth=0.4,
            zorder=20,
            s=50,
        )
        for j, x_j in enumerate(x_space):
            score = r[y].iloc[j]
            m = marker_arr[j % len(marker_arr)]
            sdef = 60
            s = (
                sdef * 0.66 if m in ["s", "D"] else sdef
            )  # these two markers are unexplicably larger in default pyplot
            ax.scatter(
                x_j,
                score,
                alpha=0.8,
                marker=m,
                linewidth=0.4,
                edgecolor=(0.8, 0.8, 0.8),
                s=s,
                color=cmap((j // n_markers) + c_offs_left),
            )
            if i == 0:
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker=m,
                        color="w",
                        label=r[match_col].iloc[j], # Here it goes wrong
                        markerfacecolor=gmap((j // n_markers) + c_offs_left),
                        markersize=1.2 * np.sqrt(s),
                    )
                )
                # need to translate markersize between scatter and plt function, 1.2*sqrt() seems to work kind of
    ax.set_xticks(np.arange(1, num_x + 1))
    if x_match_sort is not None:
        x_order[x_order.index(x_match_sort)] = sort_marker + x_order[x_order.index(x_match_sort)]
    ax.set_xticklabels(x_order)
    ax.set_xlim(0, num_x + 1)
    for i, tick in enumerate(ax.get_xticklabels()):
        if len(tick.get_text()) > 0:
            tick.set_ha("right")
            tick.set_rotation_mode("anchor")
            tick.set_rotation(10)
            tick.set_color(cp[i])
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return ax, legend_handles, legend_labels, [Line2D([0], [0], marker='X', color='black', markersize=6, linestyle='None')]