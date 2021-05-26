# -*- coding: utf-8 -*-
"""
Created on Wed May 12 16:07:41 2021
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, kurtosis, mannwhitneyu


path = r'./spot.csv'
df = pd.read_csv(path)
pipelines = df['pipeline'].unique()

fig, axes = plt.subplots(1, 3, sharex=True,sharey = True, figsize =(12,3), dpi = 200)
for p in range(pipelines.shape[0]):
    scores = df[df['pipeline'] == pipelines[p]]['score'].to_numpy()
    space = np.linspace(0, 1.1, 110)
    axes[p].plot(space, norm.pdf(space,  loc = np.mean(scores), scale = np.std(scores)), color = 'tab:orange', label = 'superimposed normal distribution')
    axes[p].hist(scores, density = True, label = 'density histogram', color = 'tab:blue')
    axes[p].set_title("{}\nmean: {}  std: {}   kur: {}".format(pipelines[p], np.round(np.mean(scores), 2), np.round(np.std(scores), 2), np.round(kurtosis(scores), 2)))
# Add common lables and legend
fig.text(0.5, 0.00, 'Score', ha='center')
fig.text(0.08, 0.65, 'bin count / total count', rotation='vertical')
axes.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)

# Shift plots down for common title
fig.subplots_adjust(top=0.75)
fig.suptitle('The distribution of the scores per pipeline')
plt.show()

# Statistical tests
tsls = df[df['pipeline'] == 'Tangent space LDA shrinkage']['score'].to_numpy()
resp = df[df['pipeline'] == 'Response shrinkage']['score'].to_numpy()
prot = df[df['pipeline'] == 'Prototype shrinkage']['score'].to_numpy()

# Calculate p values
_, pvalue = mannwhitneyu(tsls, resp, alternative = 'two-sided')
print("The p value from the double sided dependent samples t-test between the baseline and the response shrinkage is {}.\n".format(pvalue))

_, pvalue = mannwhitneyu(tsls, prot, alternative = 'two-sided')
print("The p value from the double sided dependent samples t-test between the baseline and the prototype shrinkage is {}.\n".format(pvalue))
