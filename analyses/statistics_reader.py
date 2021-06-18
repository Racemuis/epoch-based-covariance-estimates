# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 11:57:04 2021

Parses the statistics from main_moabb_pipeline.py
"""
import regex
import numpy as np

integers = regex.compile('^([0-9]*)'+ '\s*' +'([0-9]*)$')
doubles = regex.compile('^([0-9]*\.*[0-9]*)'+ '\s*' +'([0-9]*)$')

with open('./results/new_stats/spd_bnci_1_resp.txt') as f:
    lines = f.readlines()

first = np.zeros(len(lines))
second = np.zeros(len(lines))
for i in range(len(lines)):
    first[i] = float(doubles.match(lines[i]).group(1))
    second[i] = float(doubles.match(lines[i]).group(2))
    
print(first.shape)
print(second.shape)
avg = np.sum(first)/np.sum(second)*100
print("The percentage of non-spd matrices is {}".format(np.round(avg, 2)))