# -*- coding: utf-8 -*-
"""
Created on Thu May 27 13:56:51 2021
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from strip import plot_matched

def read_data(path, dataset):
    df = pd.read_csv(path)
    df = df.groupby(['pipeline'])['score'].aggregate([np.mean]).reset_index()
    df["dataset"] = [dataset]*df.shape[0]
    return df

spot = read_data(r'./results/spot.csv', 'Spot')
braininvaders = read_data(r'./results/braininvaders.csv', 'Braininvaders')
bnci1 = read_data(r'./results/BNCI_1.csv', 'bnci_1')
bnci2 = read_data(r'./results/BNCI_2.csv', 'bnci_2')
bncials = read_data(r'./results/BNCI_als.csv', 'bnci_als')
efpl = read_data(r'./results/efpl.csv', 'efpl')

frames = [spot, braininvaders, bnci1, bnci2, bncials, efpl]
df = pd.concat(frames)

custom_order = ['Tangent space LDA', 'Response shrinkage', 'Super-trial shrinkage']
fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi = 200)
ax, legend_handles, legend_labels, mean_handle = plot_matched(
    df,
    x="pipeline",
    y="mean",
    match_col="dataset",
    x_order=custom_order,
    x_match_sort="Tangent space LDA",
    ax=ax,
    sort_marker=''
)
first_legend = plt.legend(handles=mean_handle, labels = ['mean'],title = 'mean', loc='lower left', bbox_to_anchor=(0.825, -.2), prop={'size': 7})

# Add the legend manually to the current Axes.
plt.gca().add_artist(first_legend)
ax.legend(legend_handles, legend_labels, title="Dataset", bbox_to_anchor=(1.0, 1.035), loc="upper left")
ax.set_title("Comparison of the pipelines")
plt.yticks(np.arange(0.5, max(df['mean']) +0.05, 0.05))
plt.grid()
plt.ylabel("Mean ROC AUC")
plt.xlabel("Pipeline")
plt.show()