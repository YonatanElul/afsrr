from afsrr.data.datasets import RPeaksDataset
from afsrr.utils.utils import extract_singular_values
from afsrr.data.physionet_readers import PhysioReader
from afsrr import (
    PROCESSED_DATA_DIR,
    EXPERIMENTS_LOGS_DIR,
    ANALYSIS_LOGS_DIR,
    RAW_LTAFDB,
    RAW_THEW_DB,
    RAW_NSRDBRR,
    RAW_AFDB,
)

import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt


text_size = 20
plt.rcParams['axes.labelsize'] = text_size
plt.rcParams['axes.titlesize'] = text_size
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 10
plt.rcParams['xtick.labelsize'] = text_size
plt.rcParams['ytick.labelsize'] = text_size
plt.rcParams['font.size'] = text_size
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['legend.fontsize'] = text_size

np.random.seed(42)

seed = 111
np.random.seed(seed)
random.seed(seed)

# Define the reader
readers = (
    PhysioReader(RAW_LTAFDB, db_name='ltafdb'),
    PhysioReader(RAW_AFDB, db_name='afdb'),
    PhysioReader(RAW_NSRDBRR, db_name='nsrdbrr'),
    PhysioReader(RAW_THEW_DB, db_name='thew'),
)

trajectory_length = 2
prediction_horizon = 1
peaks_per_sample = 64
peaks_step = 16
nsr_only = True
merge_modes = True
n_hours = 3
min_per_hour = 60
beats_per_min = 85
record_length_to_use = int(n_hours * min_per_hour * beats_per_min) if nsr_only else None
nsr_from_start = False
nsr_from_middle = False
nsr_from_end = True

# Define the list of valid patients
data_dir = PROCESSED_DATA_DIR
train_dir = os.path.join(data_dir, 'Train')
val_dir = os.path.join(data_dir, 'Val')
test_dir = os.path.join(data_dir, 'Test')

# Define the model based on which to classify
# if nsr_from_start:
start_test_model_dir = glob.glob(
    os.path.join(
        EXPERIMENTS_LOGS_DIR,
        f'Test_JointRegressionModel_*{record_length_to_use}_Beats_Start*',
    )
)[-1]

# elif nsr_from_middle:
middle_test_model_dir = glob.glob(
    os.path.join(
        EXPERIMENTS_LOGS_DIR,
        f'Test_JointRegressionModel_*{record_length_to_use}_Beats_Middle*',
    )
)[-1]

# elif nsr_from_end:
end_test_model_dir = glob.glob(
    os.path.join(
        EXPERIMENTS_LOGS_DIR,
        f'Test_JointRegressionModel_*{record_length_to_use}_Beats_End*',
    )
)[-1]


exp_name = 'af_at_end'
only_compare_patients_present_in_all_lengths = True
if __name__ == "__main__":
    # Load labels
    with_labels = True
    test_ds = RPeaksDataset(
        mode='Test',
        temporal_horizon=trajectory_length,
        prediction_horizon=prediction_horizon,
        dir_path=test_dir,
        peaks_per_sample=peaks_per_sample,
        peaks_step=peaks_step,
        nsr_only=nsr_only,
        record_length_to_use=record_length_to_use,
        merge_modes=merge_modes,
        with_labels=with_labels,
    )
    test_patients = test_ds.files

    start_test_singular_values = extract_singular_values(
        path_to_model=start_test_model_dir,
        prediction_horizon_index=(0,),
        is_dmf=True,
    )
    middle_test_singular_values = extract_singular_values(
        path_to_model=end_test_model_dir,
        prediction_horizon_index=(0,),
        is_dmf=True,
    )
    end_test_singular_values = extract_singular_values(
        path_to_model=end_test_model_dir,
        prediction_horizon_index=(0,),
        is_dmf=True,
    )

    # Only take the files from patients which participated in all lengths (i.e., from 0.5 to 5 hours of sinus rhythm)
    start_relevant_af_files = (
        '04015',
        '04043',
        '04048',
        '04126',
        '04746',
        '04908',
        '05121',
        '05261',
        '06453',
        '07879',
        '08219',
        '08378',
        '08434',
    )
    start_relevant_ltaf_files = (
        '113',
        '42',
        '117',
    )

    middle_relevant_af_files = (
        '04015',
        '04043',
        '04048',
        '04126',
        '04746',
        '04908',
        '05121',
        '05261',
        '06453',
        '07879',
        '08219',
        '08378',
        '08434',
    )
    middle_relevant_ltaf_files = (
        '00',
        '113',
        '117',
        '16',
        '32',
        '42',
        '55',
        '56',
    )

    end_relevant_af_files = (
    )
    end_relevant_ltaf_files = (
        '00',
        '16',
        '32',
        '55',
        '117',
    )

    if only_compare_patients_present_in_all_lengths:
        start_test_afdb_inds = [
            i
            for i in range(len(test_patients))
            if ('afdb' in test_patients[i] and test_patients[i].split(os.sep)[-1].strip('.h5') in start_relevant_af_files)
        ]
        start_test_ltafdb_inds = [
            i
            for i in range(len(test_patients))
            if ('ltafdb' in test_patients[i] and test_patients[i].split(os.sep)[-1].strip('.h5') in start_relevant_ltaf_files)
        ]

    else:
        start_test_afdb_inds = [
            i
            for i in range(len(test_patients))
            if 'afdb' in test_patients[i].lower()
        ]
        start_test_ltafdb_inds = [
            i
            for i in range(len(test_patients))
            if 'ltafdb' in test_patients[i].lower()
        ]

    test_thew_inds = [
        i
        for i in range(len(test_patients))
        if 'thew' in test_patients[i].lower()
    ]
    start_test_afdb_sv = np.array([
        start_test_singular_values[i]
        for i in start_test_afdb_inds
    ])
    start_test_ltafdb_sv = np.array([
        start_test_singular_values[i]
        for i in start_test_ltafdb_inds
    ])
    start_test_thew_sv = np.array([
        start_test_singular_values[i]
        for i in test_thew_inds
    ])
    start_test_af_sv = np.concatenate([start_test_afdb_sv, start_test_ltafdb_sv], axis=0)
    x_axis = list(range(1, len(start_test_afdb_sv[0]) + 1))

    middle_test_afdb_inds = [
        i
        for i in range(len(test_patients))
        if ('afdb' in test_patients[i] and test_patients[i].split(os.sep)[-1].strip('.h5') in middle_relevant_af_files)
    ]
    middle_test_ltafdb_inds = [
        i
        for i in range(len(test_patients))
        if ('ltafdb' in test_patients[i] and test_patients[i].split(os.sep)[-1].strip('.h5') in middle_relevant_ltaf_files)
    ]
    middle_test_afdb_sv = np.array([
        middle_test_singular_values[i]
        for i in middle_test_afdb_inds
    ])
    middle_test_ltafdb_sv = np.array([
        middle_test_singular_values[i]
        for i in middle_test_ltafdb_inds
    ])
    middle_test_thew_sv = np.array([
        middle_test_singular_values[i]
        for i in test_thew_inds
    ])
    middle_test_af_sv = np.concatenate([middle_test_afdb_sv, middle_test_ltafdb_sv], axis=0)

    end_test_ltafdb_inds = [
        i
        for i in range(len(test_patients))
        if ('ltafdb' in test_patients[i] and test_patients[i].split(os.sep)[-1].strip('.h5') in end_relevant_ltaf_files)
    ]
    end_test_ltafdb_sv = np.array([
        end_test_singular_values[i]
        for i in start_test_ltafdb_inds
    ])
    end_test_thew_sv = np.array([
        end_test_singular_values[i]
        for i in test_thew_inds
    ])
    end_test_af_sv = end_test_ltafdb_sv

    average_start_af = start_test_af_sv.mean(axis=0)
    average_middle_af = middle_test_af_sv.mean(axis=0)
    average_end_af = end_test_af_sv.mean(axis=0)
    average_healthy = middle_test_thew_sv.mean(axis=0)

    fig, ax = plt.subplots(figsize=(21, 11))
    ax.plot(x_axis, average_start_af, color='lightcoral', marker='x', label='Post-AF Sinus Rhythm')
    ax.plot(x_axis, average_middle_af, color='royalblue', marker='d', label='Sinus Rhythm Between AF')
    ax.plot(x_axis, average_end_af, color='darkred', marker='o', label='Pre-AF Sinus Rhythm')
    ax.plot(x_axis, average_healthy, color='darkgreen', marker='s', label='Healthy')
    ax.errorbar(
        x=x_axis,
        y=average_start_af,
        yerr=average_start_af.std(axis=0), color='lightcoral', marker='x', alpha=0.6,
    )
    ax.errorbar(
        x=x_axis,
        y=average_middle_af,
        yerr=average_middle_af.std(axis=0),
        color='royalblue',
        marker='d',
        alpha=0.5,
    )
    ax.errorbar(
        x=x_axis,
        y=average_end_af,
        yerr=average_end_af.std(axis=0),
        color='darkred',
        marker='o',
        alpha=0.4,
    )
    ax.errorbar(
        x=x_axis,
        y=average_healthy,
        yerr=average_healthy.std(axis=0),
        color='darkgreen',
        marker='o',
        alpha=0.3,
    )
    ax.set_xlabel('Singular Value Index', weight='bold')
    ax.set_ylabel('Singular Value', weight='bold')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(
        fname=os.path.join(ANALYSIS_LOGS_DIR, f'singular_values_comparison.pdf'),
        orientation='landscape',
        format='pdf',
        bbox_inches='tight',
    )
    plt.savefig()
    plt.show()

