#########################################\
#   Calculates CFMs matrices from input EEG data
#   !!!!!
#   INPUT DATA
#   !!!!!
#   Expects: 3D numpy array with dimensions:
#            n_cfms_samples : electrodes : sample_data
#                       ===
#            (PARTICIPANTs * TRIALs * SEGMENTSfromtrials : ELECTODE(s) : timestamps )
#
# !!! NEED TO STORE LABELS FOR EACH SAMPLE!!!
#   Can be obtained at some point from something like:
#   4D numpy array with dimensions PARTICIPANT(s): TRIAL(s): ELECTRODE(s) : timestamps )
#########################################\
import os, glob
from PIL import Image
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl

from scipy.signal import hilbert
from pre_seg_labels import preprocess


# def get_data():
#     eeg, lab = preprocess()
#     return eeg, lab


def get_data():
    eeg, lab = preprocess()
    # Total number of electrodes (14 in this case)
    total_electrodes = eeg.shape[1]

    """
    Fz (AF3)  → 0
    C3 (FC5)  → 3
    Cz (F3)   → 2
    C4 (FC6)  → 10
    Pz (P7)   → 5
    PO7 (O1)  → 6
    Oz (O2)   → 7
    PO8 (P8)  → 8
    
    unicorn -> dreamer
    Fz → AF3
    C3 → FC5
    Cz → F3
    C4 → FC6
    Pz → P7
    PO7 → O1
    Oz → O2
    PO8 → P8
    """
    num_samples = eeg.shape[0]
    num_electrodes = eeg.shape[1]
    num_timepoints = eeg.shape[2]

    # Indices of the selected electrodes
    selected_indices = [0, 3, 2, 10, 5, 6, 7, 8]

    selected_eeg_data = eeg[:, selected_indices, :]
    # EEG: (20884, 8, 1024)
    return selected_eeg_data, lab


def correlation_cfm(one_sample: np.ndarray) -> np.ndarray:
    """
    Computes correlation between two signals from 2 electrodes for all combinations of electrodes
    Expects to be working on one layer of eeg at a time
        -> electrode : samples(ts)
    """
    # takes correlations between rows
    correl = np.corrcoef(one_sample)
    return correl.round(3)


def phase_lock_cfm(one_sample: np.ndarray, sampling_frequency: int = 128) -> np.ndarray:
    """ ""
    Calculate phase-locking value as a measure of synchronization between 2 signals
    Expects to be working on one layer of eeg at a time
        -> electrode : samples(ts)

    Produces nelectrodes x nelectrodes Connectivity Feature Map for one layer of eeg
    """
    # time
    total_time = one_sample.shape[-1] * 1 / sampling_frequency

    # Assuming signal_A and signal_B are your time series signals
    analytic_signals = hilbert(one_sample, axis=1)

    # Extract instantaneous phases
    phases = np.angle(analytic_signals)

    # vectorized PLV calculation for all electrodes
    # phase differences between all phases at all time points
    phase_diffs = phases[:, np.newaxis, :] - phases[np.newaxis, :, :]
    complex_phase_diffs = np.exp(1j * phase_diffs)
    # average over timepoints
    PLV_cfm = np.abs(np.mean(complex_phase_diffs, axis=-1))

    return PLV_cfm


def plv_formula(phases1, phases2):
    return np.abs(np.mean([np.exp(1j * (p1 - p2)) for p1, p2 in zip(phases1, phases2)]))


def construct_cfms(eeg, cfm_method1, cfm_method2):
    """'
    Fusion in systematic way:
    top of diagonal = cfm1; below diagonal = cfm2
    apply min-max scaling to have same values? maybe not necessary
    """
    cfm1, cfm2 = zip(
        *[
            (cfm_method1(eeg[s, :, :]), cfm_method2(eeg[s, :, :]))
            for s in range(eeg.shape[0])
        ]
    )
    cfm1 = np.triu(np.array(cfm1))
    cfm2 = np.tril(np.array(cfm2))

    # fuse arrays
    cfm_fused = cfm1 + cfm2
    return cfm_fused


def main(write=True):
    # Export the images to folder
    os.makedirs("./data_og/imgs/", exist_ok=True)
    os.makedirs("./data_og/raw/", exist_ok=True)
    if write:
        # 2D label with arousal & valence for each CFM
        eeg, labels = get_data()
        cfms = construct_cfms(eeg, correlation_cfm, phase_lock_cfm)

        with open("./data_og/raw/eeg.pkl", "wb") as file_:
            pkl.dump(eeg, file_)

        with open("./data_og/raw/labels.pkl", "wb") as file_:
            pkl.dump(labels, file_)

        with open("./data_og/raw/cfms.pkl", "wb") as file_:
            pkl.dump(cfms, file_)
    else:
        with open("./data_og/raw/eeg.pkl", "rb") as file_:
            eeg = pkl.load(file_)

        with open("./data_og/raw/labels.pkl", "rb") as file_:
            labels = pkl.load(file_)

        with open("./data_og/raw/cfms.pkl", "rb") as file_:
            cfms = pkl.load(file_)

    # Store the labels
    labels = np.concatenate(
        (np.expand_dims(np.arange(labels.shape[0]), axis=-1), labels), axis=1
    )
    pd.DataFrame(labels, columns=["imgs_num", "arousal", "valenca"]).to_csv(
        "./data/labels.csv", index=False
    )

    # EEG: (20884, 14, 1024)
    # CFMS: (20884, 14, 1024)
    print(f"Dimensions check: EEG {eeg.shape}, CFMS {cfms.shape} LABELS {labels.shape}")
    # sns.heatmap(cfms[np.random.randint(cfms.shape[0]), :, :])
    # plt.show()
    # breakpoint()


if __name__ == "__main__":
    main(write=True)
