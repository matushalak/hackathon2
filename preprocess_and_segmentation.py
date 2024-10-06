# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 12:42:53 2024

@author: antreasvasileiou , olejurgensen
"""

import mne
import numpy as np
from scipy.io import loadmat
import os

def preprocess():
    dpath = os.getcwd() + '/DREAMER.mat'
    mat = loadmat(dpath)
    DREAMER = mat['DREAMER'][0]

    # DREAMER Information
    sr = DREAMER['EEG_SamplingRate'][0][0,0]
    electrodes = [electrode[0] for electrode in DREAMER['EEG_Electrodes'][0][0]]
    num_subjects = DREAMER['noOfSubjects'][0][0,0]
    num_films = DREAMER['noOfVideoSequences'][0][0,0]

    # DREAMER Dictionary
    DREAMER_info = {
        'SamplingRate': sr,
        'Electrodes': electrodes,
        'NumOfSubjects': num_subjects,
        'NumOfVideos': num_films
    }

    data = DREAMER['Data'][0][0] # EEG data

    subjects_data = [] # to append EEG data

    for subject in data:
        subject_info = {
            'Age': subject[0,0]['Age'][0],
            'Gender': subject[0,0]['Gender'][0],
            'ScoreValence': subject[0,0]['ScoreValence'].ravel(),
            'ScoreArousal': subject[0,0]['ScoreArousal'].ravel(),
            'ScoreDominance': subject[0,0]['ScoreDominance'].ravel(),
            'EEG': []
        }

        for film in range(num_films):
            film_data = []

            for elec in range(len(electrodes)):
                baseline_data = subject[0,0]['EEG']['baseline'][0,0][film][0][:,elec]
                stimuli_data = subject[0,0]['EEG']['stimuli'][0,0][film][0][:,elec]

                film_data.append({
                    'baseline': baseline_data,
                    'stimuli': stimuli_data
                })

            subject_info['EEG'].append(film_data)
        subjects_data.append(subject_info)

    DREAMER_info.update({'SubjectsData': subjects_data}) # append to DREAMER_info

    #%% Read with mne - Preprocessing
    sampling_rate = DREAMER_info['SamplingRate']
    electrode_names = DREAMER_info['Electrodes']

    data_clean = []

    for subject_idx, subject_data in enumerate(DREAMER_info['SubjectsData']):
        print(f'''

            Subject {subject_idx+1}/{DREAMER_info['NumOfSubjects']}

            ''')

        subject_clean = []

        for film_idx, film_data in enumerate(subject_data['EEG']):
            print(f'''

                Film {film_idx+1}/{num_films} - Subject {subject_idx+1}

                ''')
            eeg_data = np.array([elec_data['stimuli'] for elec_data in film_data])
            eeg_data = eeg_data / 1e6
            info = mne.create_info(ch_names=electrode_names, sfreq=sampling_rate, ch_types='eeg')
            raw = mne.io.RawArray(eeg_data, info)

            # Preprocess
            raw_clean = raw.copy().filter(l_freq=30, h_freq=49)  # gamma-band filter
            # raw_clean.notch_filter(freqs=[50])  # notch filter for 50 Hz powerline noise
            raw_clean.set_eeg_reference('average', projection=True)  # re-reference to reduce noise

            subject_clean.append(raw_clean)
        data_clean.append(subject_clean)

    #%%
    # requires subject_data and data_clean from preprocessing code
    cfm_array = []

    for subject in range(len(data_clean)): 
        subject_data = []
        
        for film in range(len(data_clean[subject])): 
            epochs = mne.make_fixed_length_epochs(data_clean[subject][film], 
                                                        duration=8, preload=False, overlap = 4)
            # array
            epoch_data = epochs.get_data()
            if epoch_data.shape[-2:] != (14,1024):
                breakpoint()

            cfm_array.append(epoch_data)

    cfm_array = np.vstack(cfm_array)

    return cfm_array
    # (PARTICIPANTs : TRIALs (movies) : SEGMENTSfromtrials : ELECTODE(s) : timestamps )

    # not sure if the data structure is optimal, can't get it into a fully array because the number of
    # segemngts are different for each trial
    # also does not include labels yet