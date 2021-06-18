# -*- coding: utf-8 -*-
"""
Created on Wed May 12 16:07:41 2021

Computes the wilcoxon signed rank test of the results.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, kurtosis, wilcoxon

plt.style.use('default') 

import scipy
print(scipy.__version__)

path = r'./results/bnci_1.csv'
df = pd.read_csv(path)
pipelines = df['pipeline'].unique()

fig, axes = plt.subplots(1, 3, sharex=True,sharey = True, figsize =(12,3), dpi = 200)
colours = ['tab:blue', 'tab:orange', 'tab:green']
for p in range(pipelines.shape[0]):
    scores = df[df['pipeline'] == pipelines[p]]['score'].to_numpy()
    space = np.linspace(0, 1.1, 110)
    axes[p].plot(space, norm.pdf(space,  loc = np.mean(scores), scale = np.std(scores)), color = 'tab:orange', label = 'superimposed normal distribution')
    axes[p].hist(scores, density = True, label = 'density histogram', color = 'tab:blue')
    axes[p].set_title("{}\nmean: {}  std: {}   kur: {}".format(pipelines[p], np.round(np.mean(scores), 2), np.round(np.std(scores), 2), np.round(kurtosis(scores), 2)), color = colours[p])
    axes[p].grid()
    
# Add common lables and legend
fig.text(0.5, -0.20, 'ROC AUC score', ha='center')
axes.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)

# Shift plots down for common title
fig.subplots_adjust(top=0.70)
fig.suptitle('The distribution of the scores per pipeline\nBNCI_1')

axes[0].set_ylabel("Normalized count")

plt.show()

# Statistical tests
tsls_1 = df[df['pipeline'] == 'Tangent space LDA']['score'].to_numpy()
resp_1 = df[df['pipeline'] == 'Response shrinkage']['score'].to_numpy()
prot_1 = df[df['pipeline'] == 'Super trial shrinkage']['score'].to_numpy()


check = df.groupby(['pipeline', 'subject'])['score'].aggregate([np.mean]).reset_index()

tsls = check[check['pipeline'] == 'Tangent space LDA']['mean'].to_numpy()
resp = check[check['pipeline'] == 'Response shrinkage']['mean'].to_numpy()
prot = check[check['pipeline'] == 'Super-trial shrinkage']['mean'].to_numpy()

print(tsls_1.shape)

# Calculate p values
_, pvalue = wilcoxon(tsls, resp, alternative= "two-sided")
print("The p value from the double sided dependent samples Wilcoxon signed-rank test between the baseline and the response shrinkage is {}.\n".format(pvalue))

_, pvalue = wilcoxon(tsls, prot, alternative= "two-sided")
print("The p value from the double sided dependent samples Wilcoxon signed-rank test between the baseline and the prototype shrinkage is {}.\n".format(pvalue))
