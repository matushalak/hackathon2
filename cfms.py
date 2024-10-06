import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import hilbert, resample_poly, butter, filtfilt
from preprocess_and_segmentation import preprocess
import os
import time

# Updated CFMS code

def downsample_to128 (eeg):
    '''
    Filters out noise to only include gamma frequency band
    '''
    return resample_poly(eeg, up = 64, down = 125, axis = -1)

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=30, highcut=50, fs=250, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # Apply the filter along the last axis (axis=-1)
    y = filtfilt(b, a, data, axis=-1)
    return y

def get_data():
    # eeg = preprocess()
    eeg = np.load('raw_data1.npy').T
    eeg = np.reshape(eeg, (1, eeg.shape[0], -1))
    gamma_eeg = bandpass_filter(eeg) # gamma filter
    downsampled_eeg = downsample_to128(gamma_eeg)
    return downsampled_eeg

def correlation_cfm(one_sample:np.ndarray)->np.ndarray:
    '''
    Computes correlation between two signals from 2 electrodes for all combinations of electrodes
    Expects to be working on one layer of eeg at a time
        -> electrode : samples(ts) 
    '''
    # takes correlations between rows
    correl = np.corrcoef(one_sample)
    return correl.round(3)

def phase_lock_cfm(one_sample:np.ndarray, sampling_frequency:int =128)->np.ndarray:
    """""
    Calculate phase-locking value as a measure of synchronization between 2 signals
    Expects to be working on one layer of eeg at a time
        -> electrode : samples(ts) 

    Produces nelectrodes x nelectrodes Connectivity Feature Map for one layer of eeg
    """
    # time
    total_time = one_sample.shape[-1] * 1/sampling_frequency
    
    # Assuming signal_A and signal_B are your time series signals
    analytic_signals= hilbert(one_sample, axis = 1)

    # Extract instantaneous phases
    phases = np.angle(analytic_signals)

    # vectorized PLV calculation for all electrodes
    # phase differences between all phases at all time points
    phase_diffs = phases[:, np.newaxis, :] - phases[np.newaxis, :, :]
    complex_phase_diffs = np.exp(1j * phase_diffs)
    # average over timepoints
    PLV_cfm = np.abs(np.mean(complex_phase_diffs, axis = -1))
    
    return min_max_scale_neg1_to_1(PLV_cfm)

def min_max_scale_neg1_to_1(array):
    # Find the minimum and maximum of the input array
    min_val = np.min(array)
    min_val = 0.0000001 if min_val <= 0 else min_val # avoid dividing by 0
    max_val = np.max(array)
    
    # Apply the -1 to 1 scaling formula
    scaled_array = 2 * (array - min_val) / (max_val - min_val) - 1
    
    return scaled_array

def construct_cfms(eeg, cfm_method1, cfm_method2):
    ''''
    Fusion in systematic way: 
    top of diagonal = cfm1; below diagonal = cfm2
    apply min-max scaling to have same values? maybe not necessary
    '''
    cfm1, cfm2 = zip(*[(cfm_method1(eeg[s,:,:]), cfm_method2(eeg[s,:,:])) for s in range(eeg.shape[0])])
    cfm1 = np.triu(np.array(cfm1))
    cfm2 = np.tril(np.array(cfm2))

    # fuse arrays
    cfm_fused = (cfm1 + cfm2) / 2
    return cfm_fused

def main():
    start = time.time()
    os.makedirs("./our_brain_cfms", exist_ok=True)
    eeg = get_data()
    cfms = construct_cfms(eeg, correlation_cfm, phase_lock_cfm)

    num_cfm = sum([1 if 'cfm' in f else 0 for f in os.listdir('our_brain_cfms')])
    np.save(f'./our_brain_cfms/cfm{num_cfm+1}.npy', cfms)
    # EEG: (20884, 14, 1024)
    # CFMS: (20884, 14, 1024)
    print(f'Dimensions check: EEG {eeg.shape}, CFMS {cfms.shape}')
    # sns.heatmap(cfms[np.random.randint(cfms.shape[0]),:,:])
    sns.heatmap(cfms[0,:,:])
    plt.show()
    print(time.time() - start)
    breakpoint()

if __name__ == "__main__":
    main()