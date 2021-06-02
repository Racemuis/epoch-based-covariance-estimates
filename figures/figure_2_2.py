# -*- coding: utf-8 -*-
"""
Created on Sun May  2 18:16:28 2021

Figure 2.2: An aggregated ERP response with a soa of 226.
Code partially taken from SpotPilotData (adapted).
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
import yaml
import glob

import os
os.chdir('..')

from methods.datasets.spot_pilot import SpotPilotData

mne.set_log_level(False)

def preprocessed_to_epoch(preprocessed_data, decimate=10, baseline_ival=(-.2, 0)):
    raw_stim_ids = {"Stimulus/S  1": 1, "Stimulus/S 21": 2} 
    class_ids = {"Target": 2, "Non-target": 1}
    reject = dict()
    events = mne.events_from_annotations(preprocessed_data, event_id=raw_stim_ids)[0]
    epo_data = mne.Epochs(preprocessed_data, events, event_id=class_ids,
                          baseline=baseline_ival, decim=decimate,
                          reject=reject, proj=False, preload=True)
    return epo_data


def _data_path(subject, path):
        path_folder = path + f'/subject{subject}'
        
        # get the path to all files
        pattern = r'/*Run_2*.vhdr'
        subject_paths = glob.glob(
            path_folder + pattern)
        return sorted(subject_paths)

def average_single_subject_data(subject, path, load_single_trials, plot_channels):
        """return data for a single subject"""
        getter = SpotPilotData(load_single_trials=True, reject_non_iid=True)
        file_path_list = _data_path(subject, path)
        evo_t = np.zeros(( len(plot_channels), 71))
        evo_nt = np.zeros((len(plot_channels),71))
        soa_t = 0
        times = []
        for p_i, file_path in enumerate(file_path_list):
            file_exp_info = SpotPilotData._filename_trial_info_extraction(file_path)
            soa = file_exp_info['soa']
            # SOA selected by hand, need to verify whether it's in there
            if soa != 226:
                continue
            soa_t += 1
            data = getter._get_single_run_data(file_path)
            
            data.filter(0.5, 16, method='iir')
            epo_data = preprocessed_to_epoch(data, baseline_ival=(-0.2, 0))
            
            # Compute target and non-target ERP at channels
            evo_t += epo_data['Target'].average(picks=plot_channels).data
            evo_nt += epo_data['Non-target'].average(picks=plot_channels).data
            times = epo_data['Non-target'].average(picks=plot_channels).times
            target = epo_data['Target'].get_data().shape[0]
            non_target = epo_data['Non-target'].get_data().shape[0]
        print('Averaged over {} target responses and {} non-target responses. '.format(soa_t * target, soa_t * non_target))
        return evo_t/soa_t, evo_nt/soa_t, times
    
        
# Extract configurations  
local_cfg_file = r'local_config.yaml'
analysis_cfg_file = r'analysis_config.yaml'
with open(local_cfg_file, 'r') as conf_f:
    local_cfg = yaml.load(conf_f, Loader=yaml.FullLoader)    
with open(analysis_cfg_file, 'r') as conf_f:
    ana_cfg = yaml.load(conf_f, Loader=yaml.FullLoader)


prepro_cfg = ana_cfg['default']['data_preprocessing']
data_path = local_cfg['data_root']

# Plotting
plot_channels = ['Cz', 'Fz']
channel_styles = ['-', '--']

evo_t, evo_nt, times = average_single_subject_data(1, r'C:\Users\CC114402\Documents\Chiara\Artificial Intelligence\Leerjaar 3\Semester 2\Bachelor Thesis\Data & MOABB\Data', True, plot_channels)


micro = 10**6
# Plot target and non-target ERP for channel
for ch_i, ch in enumerate(plot_channels):
    plt.plot(times, evo_t[ch_i, :]*micro,
            linestyle=channel_styles[ch_i], color="tab:orange", label=f'{ch} Target')
    plt.plot(times, evo_nt[ch_i, :]*micro,
            linestyle=channel_styles[ch_i], color="tab:blue", label=f'{ch} Non-target')

# Define axes
plt.rcParams["figure.figsize"] = (9,6)
plt.rcParams["figure.dpi"] = 400
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (µV)')
plt.grid()
plt.title("The average ERP measured at channels Cz and Fz")
plt.legend()
plt.show()