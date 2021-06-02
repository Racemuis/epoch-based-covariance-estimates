# -*- coding: utf-8 -*-
"""
Figure 5.x: A distribution of the scores from the 3 pipelines.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rcParams

from strip import plot_matched

rcParams['font.family'] = 'sans-serif'
rcParams['font.serif'] = ['Computer Modern Roman']

path = r'./results/spot.csv'
dataframe = pd.read_csv(path)


fig, ax = plt.subplots(1, 1, facecolor='white', figsize=(6, 4), dpi = 200)

# Aggregate data
results = dataframe.groupby(['subject', 'pipeline']).aggregate([np.mean]).reset_index()

# Plot results
results.columns = [x[0] for x in results.columns]
custom_order = ['Tangent space LDA', 'Response shrinkage', 'Super-trial shrinkage']
ax, leg_handles, leg_labels, mean_handle= plot_matched(data=results, x='pipeline', y='score', x_order = custom_order, match_col = 'subject',
                                       figsize=(6, 4), x_match_sort = 'Tangent space LDA',  ax = ax, sort_marker = '')

leg_labels = leg_labels.astype(str).to_list() #0.85 for braininvaders
first_legend = plt.legend(handles=mean_handle, labels = ['mean'],title = 'mean', loc='lower left', bbox_to_anchor=(0.85, -.16), prop={'size': 7})

# Add the legend manually to the current Axes.
plt.gca().add_artist(first_legend)

# Create another legend - if you have got a few participants, set ncol = 1, else 2. 
plt.legend(handles=leg_handles, labels=leg_labels, title='Subj.', ncol=1, prop={'size': 8},
           loc='center left', bbox_to_anchor=(1, 0.5))
    
ax.set_ylabel('ROC AUC')
ax.set_xlabel('Classification method')
title = r'BNCI_als'
ax.set_title(title)
fig = ax.figure
fig.tight_layout()
plt.grid(axis = 'both')
plt.show()
    
