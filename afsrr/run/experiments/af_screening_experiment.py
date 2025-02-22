from afsrr.utils.defaults import OTHER_KEY
from afsrr.data.datasets import RPeaksDataset
from sklearn.neural_network import MLPClassifier
from afsrr.data.physionet_readers import PhysioReader
from afsrr.utils.utils import extract_singular_values, estimate_threshold
from afsrr import (
    ANALYSIS_LOGS_DIR,
    EXPERIMENTS_LOGS_DIR,
    PROCESSED_DATA_DIR,
    RAW_AFDB,
    RAW_LTAFDB,
    RAW_THEW_DB,
    RAW_NSRDBRR,
)

import os
import glob
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sk_metrics

seed = 111
np.random.seed(seed)
random.seed(seed)
logs_dir = EXPERIMENTS_LOGS_DIR

readers = (
    PhysioReader(db_path=RAW_LTAFDB, db_name='ltafdb'),
    PhysioReader(db_path=RAW_AFDB, db_name='afdb'),
    PhysioReader(db_path=RAW_NSRDBRR, db_name='nsrdbrr'),
    PhysioReader(db_path=RAW_THEW_DB, db_name='thew'),
)

trajectory_length = 2
prediction_horizon = 1
peaks_per_sample = 64
peaks_step = 16
nsr_only = True
n_hours = 0.5
min_per_hour = 60
beats_per_min = 85
record_length_to_use = int(n_hours * min_per_hour * beats_per_min) if nsr_only else None

# Define the list of valid patients
train_dir = os.path.join(PROCESSED_DATA_DIR, 'Train')
val_dir = os.path.join(PROCESSED_DATA_DIR, 'Val')
test_dir = os.path.join(PROCESSED_DATA_DIR, 'Test')
classifier_type = 'nn'

# Define the model based on which to classify
model_names = sorted(os.listdir(logs_dir))
train_model_path = sorted(
    glob.glob(
        os.path.join(
            logs_dir,
            'Train_JointRegressionModel*',
        )
    )
)[-1]
val_model_path = sorted(
    glob.glob(
        os.path.join(
            logs_dir,
            'Val_JointRegressionModel*',
        )
    )
)[-1]
test_model_path = sorted(
    glob.glob(
        os.path.join(
            logs_dir,
            'Test_JointRegressionModel*',
        )
    )
)[-1]

save_dir = os.path.join(
    ANALYSIS_LOGS_DIR,
    f"AF_Classifier_Test_{'NSR_Only_' if nsr_only else ''}{record_length_to_use}_Beats"
)
os.makedirs(save_dir, exist_ok=True)

rebalance = False
estimate_optimal_thresholds = True
alpha = 0.05
beta = None
if __name__ == "__main__":
    # Define a binary classifier
    classifier = MLPClassifier(
        hidden_layer_sizes=(16, 16, 16),
        activation='relu',
        verbose=False,
        validation_fraction=0.1,
        solver='lbfgs',
        max_iter=50000,
        alpha=0.1,
        max_fun=100000,
        n_iter_no_change=50,
        tol=1e-5,
        random_state=42,
        learning_rate='adaptive',
    )

    # Load labels
    train_ds = RPeaksDataset(
        mode='Train',
        temporal_horizon=trajectory_length,
        prediction_horizon=prediction_horizon,
        dir_path=train_dir,
        peaks_per_sample=peaks_per_sample,
        peaks_step=peaks_step,
        nsr_only=False,
    )
    val_ds = RPeaksDataset(
        mode='Val',
        temporal_horizon=trajectory_length,
        prediction_horizon=prediction_horizon,
        dir_path=val_dir,
        peaks_per_sample=peaks_per_sample,
        peaks_step=peaks_step,
        nsr_only=False,
    )
    test_ds = RPeaksDataset(
        mode='Test',
        temporal_horizon=trajectory_length,
        prediction_horizon=prediction_horizon,
        dir_path=test_dir,
        peaks_per_sample=peaks_per_sample,
        peaks_step=peaks_step,
        nsr_only=nsr_only,
        record_length_to_use=record_length_to_use,
    )

    train_labels = train_ds[0][OTHER_KEY]['af_labels'].numpy()
    val_labels = val_ds[0][OTHER_KEY]['af_labels'].numpy()
    test_labels = test_ds[0][OTHER_KEY]['af_labels'].numpy()

    # Initialize the training/testing stats
    train_stats = {
        'acc': None,
        'auc': None,
        'precision': None,
        'recall': None,
        'specificity': None,
        'f1': None,
        'confusion_matrix': None,
    }
    val_stats = train_stats.copy()
    test_stats = train_stats.copy()

    # Load singular values
    train_singular_values = extract_singular_values(
        path_to_model=train_model_path,
        prediction_horizon_index=(0,),
        is_dmf=True,
    )
    val_singular_values = extract_singular_values(
        path_to_model=val_model_path,
        prediction_horizon_index=(0,),
        is_dmf=True,
    )
    test_singular_values = extract_singular_values(
        path_to_model=test_model_path,
        prediction_horizon_index=(0,),
        is_dmf=True,
    )

    # Fit the training set
    label_freq = np.mean(train_labels)
    train_rhythms_counter = np.sum(train_labels)
    if label_freq > 0:
        balancing_ratio = int(1 / label_freq)
        if balancing_ratio > 1 and rebalance:
            positive_indices = np.where(train_labels == 1)[0]
            augmented_labels = np.ones((balancing_ratio - 1) * len(positive_indices))
            augmented_samples = np.repeat(
                train_singular_values[positive_indices],
                repeats=(balancing_ratio - 1),
                axis=0,
            )
            samples = np.concatenate([train_singular_values, augmented_samples], axis=0)
            samples_labels = np.concatenate([train_labels, augmented_labels], axis=0)

        else:
            samples = train_singular_values
            samples_labels = train_labels

    else:
        samples = train_singular_values
        samples_labels = train_labels

    # Fit the model on the r-th rhythm
    train_rhythms_freqs = label_freq
    fitted_model = classifier.fit(samples, samples_labels)
    train_probs = fitted_model.predict_proba(train_singular_values)
    predicted_labels = fitted_model.predict(train_singular_values)
    train_stats['acc'] = sk_metrics.accuracy_score(train_labels, predicted_labels)

    if 1 > label_freq > 0:
        train_stats['auc'] = sk_metrics.roc_auc_score(train_labels, train_probs[:, 1])

    else:
        train_stats['auc'] = np.nan

    if label_freq > 0:
        train_stats['precision'] = sk_metrics.precision_score(train_labels, predicted_labels)
        train_stats['recall'] = sk_metrics.recall_score(train_labels, predicted_labels)
        train_stats['f1'] = sk_metrics.f1_score(train_labels, predicted_labels)

    else:
        train_stats['precision'] = np.nan
        train_stats['recall'] = np.nan
        train_stats['f1'] = np.nan

    # Compute specificity
    confusion_matrix = sk_metrics.confusion_matrix(
        y_true=train_labels,
        y_pred=predicted_labels,
        normalize='all',
    ).ravel()
    if len(confusion_matrix) < 4:
        tn = np.sum(((train_labels == 0) * (predicted_labels == 0))) / train_labels.size
        fp = np.sum(((train_labels == 0) * (predicted_labels == 1))) / train_labels.size
        fn = np.sum(((train_labels == 1) * (predicted_labels == 0))) / train_labels.size
        tp = np.sum(((train_labels == 1) * (predicted_labels == 1))) / train_labels.size
        confusion_matrix = (tn, fp, fn, tp)

    specificity = confusion_matrix[0] / (confusion_matrix[0] + confusion_matrix[1])
    train_stats['specificity'].append(specificity)
    train_stats['confusion_matrix'].append(confusion_matrix)

    # Evaluate on the validation set
    label_freq = np.mean(val_labels)
    val_rhythms_counter = np.sum(val_labels)
    val_rhythms_freqs = label_freq
    val_probs = fitted_model.predict_proba(val_singular_values)

    if estimate_optimal_thresholds:
        optimal_threshold = estimate_threshold(
            probs=val_probs[:, 1],
            labels=val_labels,
            alpha=alpha,
            beta=beta,
        )
        print(f"Optimal threshold: {optimal_threshold}")
        predicted_labels = val_probs[:, 1]
        predicted_labels[predicted_labels >= optimal_threshold] = 1
        predicted_labels[predicted_labels < optimal_threshold] = 0

    else:
        predicted_labels = fitted_model.predict(val_singular_values)
        optimal_threshold = 0.5

    val_stats['acc'] = sk_metrics.accuracy_score(val_labels, predicted_labels)
    if 1 > label_freq > 0:
        val_stats['auc'] = sk_metrics.roc_auc_score(val_labels, val_probs[:, 1])

    else:
        val_stats['auc'] = np.nan

    if label_freq > 0:
        val_stats['precision'] = sk_metrics.precision_score(val_labels, predicted_labels)
        val_stats['recall'] = sk_metrics.recall_score(val_labels, predicted_labels)
        val_stats['f1'] = sk_metrics.f1_score(val_labels, predicted_labels)

    else:
        val_stats['precision'] = np.nan
        val_stats['recall'] = np.nan
        val_stats['f1'] = np.nan

    confusion_matrix = sk_metrics.confusion_matrix(
        y_true=val_labels,
        y_pred=predicted_labels,
        normalize='all',
    ).ravel()
    if len(confusion_matrix) < 4:
        tn = np.sum(((test_labels == 0) * (predicted_labels == 0))) / test_labels.size
        fp = np.sum(((test_labels == 0) * (predicted_labels == 1))) / test_labels.size
        fn = np.sum(((test_labels == 1) * (predicted_labels == 0))) / test_labels.size
        tp = np.sum(((test_labels == 1) * (predicted_labels == 1))) / test_labels.size
        confusion_matrix = (tn, fp, fn, tp)

    specificity = confusion_matrix[0] / (confusion_matrix[0] + confusion_matrix[1])
    val_stats['specificity'].append(specificity)
    val_stats['confusion_matrix'].append(confusion_matrix)

    # Evaluate on the test set
    label_freq = np.mean(test_labels)
    test_rhythms_counter = np.sum(test_labels)
    test_rhythms_freqs = label_freq
    test_probs = fitted_model.predict_proba(test_singular_values)

    if estimate_optimal_thresholds:
        predicted_labels = test_probs[:, 1].copy()
        predicted_labels[predicted_labels >= optimal_threshold] = 1
        predicted_labels[predicted_labels < optimal_threshold] = 0

    else:
        predicted_labels = fitted_model.predict(test_singular_values)

    test_stats['acc'] = sk_metrics.accuracy_score(test_labels, predicted_labels)
    if 1 > label_freq > 0:
        test_stats['auc'] = sk_metrics.roc_auc_score(test_labels, test_probs[:, 1])

    else:
        test_stats['auc'] = np.nan

    if label_freq > 0:
        test_stats['precision'] = sk_metrics.precision_score(test_labels, predicted_labels)
        test_stats['recall'] = sk_metrics.recall_score(test_labels, predicted_labels)
        test_stats['f1'] = sk_metrics.f1_score(test_labels, predicted_labels)

    else:
        test_stats['precision'] = np.nan
        test_stats['recall'] = np.nan
        test_stats['f1'] = np.nan

    tn = np.sum(((test_labels == 0) * (predicted_labels == 0))) / test_labels.size
    fp = np.sum(((test_labels == 0) * (predicted_labels == 1))) / test_labels.size
    fn = np.sum(((test_labels == 1) * (predicted_labels == 0))) / test_labels.size
    tp = np.sum(((test_labels == 1) * (predicted_labels == 1))) / test_labels.size
    confusion_matrix = (tn, fp, fn, tp)

    specificity = confusion_matrix[0] / (confusion_matrix[0] + confusion_matrix[1])
    test_stats['specificity'].append(specificity)
    test_stats['confusion_matrix'].append(confusion_matrix)

    roc_curve = sk_metrics.RocCurveDisplay.from_estimator(
        fitted_model,
        test_singular_values,
        test_labels,
    )
    plt.savefig(os.path.join(save_dir, f'results_{beta}_beta_{alpha}_alpha_roc.pdf'))

    with open(os.path.join(save_dir, f'results_{beta}_beta_{alpha}_alpha.pkl'), 'wb') as f:
        pickle.dump(
            obj={
                'train_stats': train_stats,
                'train_rhythms_freqs': train_rhythms_freqs,
                'train_rhythms_counter': train_rhythms_counter,
                'val_stats': val_stats,
                'val_rhythms_freqs': val_rhythms_freqs,
                'val_rhythms_counter': val_rhythms_counter,
                'test_stats': test_stats,
                'test_rhythms_freqs': test_rhythms_freqs,
                'test_rhythms_counter': test_rhythms_counter,
                'trained_model': fitted_model,
                'x_test': test_singular_values,
                'y_test': test_labels,
                'y_test_hat': test_probs[:, 1],
                'optimal_threshold': optimal_threshold,
            },
            file=f,
        )

    print(f"test_stats: {test_stats}")
