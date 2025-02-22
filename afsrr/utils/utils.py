from typing import Sequence, Union
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from afsrr.data.physionet_readers import PhysioReader
from typing import Sequence, Dict, Any, Callable, Optional, Tuple, Union, List
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    euclidean_distances,
)

import os
import torch
import pickle
import numpy as np
import sklearn.metrics as sk_metrics


RHYTHMS_CODINGS = {
    0: 'NSR',
    1: 'LP-NSR',
    2: 'NA',
    3: 'AF',
    4: 'SVT',
    5: 'VT',
    6: 'VBig',
    7: 'VTrig',
    8: 'IR',
    9: 'ABig',
    10: 'SBrady',
    11: '2° Heart Block',
    12: 'Pre-Excitation (WPW)',
    13: 'Atrial Flutter',
    14: 'Paced Rhythm',
    15: 'Nodal (A-V Junctional) Rhythm',
    16: 'Junctional Rhythm',
    17: 'Ventricular Flutter',
    18: 'Ventricular Fibrillation',
}


def load_dynamics_operators(
        path_to_model: str,
) -> (Sequence[np.ndarray], Sequence[np.ndarray]):
    # Load trained model
    model_ckpt_path = os.path.join(path_to_model, 'BestModel.PyTorchModule')
    model_ckp = torch.load(model_ckpt_path)['model']

    # Extract dynamics components
    As = [
        model_ckp[m].detach().cpu().numpy()
        for m in model_ckp
        if 'U_' in m
    ]
    Bs = [
        model_ckp[m].detach().cpu().numpy()
        for m in model_ckp
        if 'S_' in m
    ]
    Cs = [
        model_ckp[m].detach().cpu().numpy().T
        for m in model_ckp
        if 'V_' in m
    ]

    dynamics = []
    for i, b in enumerate(Bs):
        if len(As) == 1:
            dynamics.append(((As[0] @ b) @ Cs[0]))

        else:
            dynamics.append(((As[i] @ b) @ Cs[i]))

    return dynamics


def get_dmf_dynamics_operators(
        path_to_model: Union[str, Sequence[str]],
) -> (Sequence[np.ndarray], Sequence[np.ndarray]):
    if isinstance(path_to_model, str):
        koopmans = load_dynamics_operators(path_to_model)

    else:
        koopmans = []
        for p in path_to_model:
            kos = load_dynamics_operators(p)
            koopmans.extend(kos)

    singular_values = [
        np.linalg.svd(k, compute_uv=False)
        for k in koopmans
    ]

    return koopmans, singular_values


def estimate_specificity(
        y_true: Sequence,
        y_pred: Sequence,
) -> float:
    spec = confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        normalize='all',
    ).ravel()
    if len(spec) < 2:
        if 0 in y_true:
            return spec

        else:
            return 1

    else:
        spec = spec[0] / (spec[0] + spec[1])

    return spec


def extract_singular_values(
        path_to_model: str,
        prediction_horizon_index: Tuple[int, ...] = (0,),
        is_dmf: bool = True,
        map_location: Optional[torch.device] = None
) -> np.ndarray:
    # Load trained model
    model_ckpt_path = os.path.join(path_to_model, 'BestModel.PyTorchModule')
    model_ckp = torch.load(model_ckpt_path, map_location=map_location)['model']

    ckpt_entries = list(model_ckp.keys())
    if 'midst.' in ckpt_entries[0]:
        model_ckp = {
            m[len('midst.'):]: model_ckp[m]
            for m in model_ckp
            if 'midst.' in m
        }

    # For the DMF case
    if is_dmf:
        if len(prediction_horizon_index) == 1:
            # Extract singular values
            singular_values_per_patient = {
                i: model_ckp[m].cpu().numpy()
                for i, m in enumerate(model_ckp)
                if f'S_per_t_per_m.{prediction_horizon_index[0]}.' in m
            }

            u = model_ckp[f'_U_per_t.{prediction_horizon_index[0]}'].detach().cpu().numpy()
            v_t = model_ckp[f'_V_per_t.{prediction_horizon_index[0]}'].detach().cpu().numpy().T
            singular_values_per_patient = {
                i: ((u @ singular_values_per_patient[i]) @ v_t)
                for i in singular_values_per_patient
            }
            singular_values_per_patient = {
                i: np.linalg.svd(singular_values_per_patient[i], compute_uv=False)
                for i in singular_values_per_patient
            }
            singular_values = [
                singular_values_per_patient[i][None, ...]
                for i in sorted(singular_values_per_patient.keys())
            ]
            singular_values = np.concatenate(singular_values, 0)

        else:
            # Extract singular values
            singular_values_per_patient = [
                {
                    i: model_ckp[m].cpu().numpy()
                    for i, m in enumerate(model_ckp)
                    if f'S_per_t_per_m.{phi}.' in m
                }
                for phi in prediction_horizon_index
            ]

            us = [model_ckp[f'_U_per_t.{phi}'].detach().cpu().numpy() for phi in prediction_horizon_index]
            v_ts = [model_ckp[f'_V_per_t.{phi}'].detach().cpu().numpy().T for phi in prediction_horizon_index]
            singular_values_per_patient = [
                {
                    i: ((u @ sv[i]) @ v_t)
                    for i in sv
                }
                for sv, u, v_t in zip(singular_values_per_patient, us, v_ts)
            ]
            singular_values_per_patient = [
                {
                    i: np.linalg.svd(sv[i], compute_uv=False)
                    for i in sv
                }
                for sv in singular_values_per_patient
            ]
            singular_values_per_patient = [
                np.concatenate([sv[i][None, ...] for i in sorted(sv.keys())], axis=0)
                for sv in singular_values_per_patient
            ]
            singular_values = np.concatenate(singular_values_per_patient, axis=1)

    else:
        # Extract singular values
        singular_values_per_patient = {
            i: model_ckp[m].cpu().numpy()
            for i, m in enumerate(model_ckp)
            if f'S_per_t_per_m.{prediction_horizon_index}.' in m
        }
        singular_values = [
            singular_values_per_patient[i][None, ...]
            for i in sorted(singular_values_per_patient.keys())
        ]
        singular_values = np.concatenate(singular_values, 0)

    return singular_values


def extract_dynamics_modules(
        path_to_model: str,
        prediction_horizon_index: Tuple[int, ...] = (0,),
        is_dmf: bool = True,
        map_location: Optional[torch.device] = None
) -> np.ndarray:
    # Load trained model
    model_ckpt_path = os.path.join(path_to_model, 'BestModel.PyTorchModule')
    model_ckp = torch.load(model_ckpt_path, map_location=map_location)['model']

    ckpt_entries = list(model_ckp.keys())
    if 'midst.' in ckpt_entries[0]:
        model_ckp = {
            m[len('midst.'):]: model_ckp[m]
            for m in model_ckp
            if 'midst.' in m
        }

    # For the DMF case
    if is_dmf:
        if len(prediction_horizon_index) == 1:
            # Extract singular values
            singular_values_per_patient = {
                i: model_ckp[m].cpu().numpy()
                for i, m in enumerate(model_ckp)
                if f'S_per_t_per_m.{prediction_horizon_index[0]}.' in m
            }

            u = model_ckp[f'_U_per_t.{prediction_horizon_index[0]}'].detach().cpu().numpy()
            v_t = model_ckp[f'_V_per_t.{prediction_horizon_index[0]}'].detach().cpu().numpy().T
            dynamics_modules = {
                i: ((u @ singular_values_per_patient[i]) @ v_t)
                for i in singular_values_per_patient
            }
            dynamics_modules = [
                dynamics_modules[i][None, ...]
                for i in sorted(dynamics_modules.keys())
            ]
            dynamics_modules = np.concatenate(dynamics_modules, 0)

        else:
            # Extract singular values
            singular_values_per_patient = [
                {
                    i: model_ckp[m].cpu().numpy()
                    for i, m in enumerate(model_ckp)
                    if f'S_per_t_per_m.{phi}.' in m
                }
                for phi in prediction_horizon_index
            ]

            us = [model_ckp[f'_U_per_t.{phi}'].detach().cpu().numpy() for phi in prediction_horizon_index]
            v_ts = [model_ckp[f'_V_per_t.{phi}'].detach().cpu().numpy().T for phi in prediction_horizon_index]
            dynamics_module_per_patient = [
                {
                    i: ((u @ sv[i]) @ v_t)
                    for i in sv
                }
                for sv, u, v_t in zip(singular_values_per_patient, us, v_ts)
            ]
            dynamics_module_per_patient = [
                np.concatenate([sv[i][None, ...] for i in sorted(sv.keys())], axis=0)
                for sv in dynamics_module_per_patient
            ]
            dynamics_modules = np.concatenate(dynamics_module_per_patient, axis=1)

    else:
        # Extract singular values
        dynamics_modules = {
            i: model_ckp[m].cpu().numpy()
            for i, m in enumerate(model_ckp)
            if f'S_per_t_per_m.{prediction_horizon_index}.' in m
        }
        dynamics_modules = [
            dynamics_modules[i][None, ...]
            for i in sorted(dynamics_modules.keys())
        ]
        dynamics_modules = np.concatenate(dynamics_modules, 0)

    return dynamics_modules


def compute_singular_values(
        path_to_model: str,
) -> np.ndarray:
    # Load trained model
    model_ckpt_path = os.path.join(path_to_model, 'BestModel.PyTorchModule')
    model_ckp = torch.load(model_ckpt_path)['model']

    # Extract singular values
    dynamics_per_patient = [
        model_ckp[m].cpu().numpy()
        for m in model_ckp
        if 'dynamics' in m and 'backdynamics' not in m
    ]
    singular_values_per_patient = [
        np.linalg.svd(d, compute_uv=False)[None, ...]
        for d in dynamics_per_patient
    ]

    singular_values = np.concatenate(singular_values_per_patient, 0)

    return singular_values


def extract_patients_rhythms(
        data_dir: str,
        readers: Sequence[PhysioReader],
) -> Dict[int, np.ndarray]:
    readers_records = {
        i: reader.reader.records
        for i, reader in enumerate(readers)
    }
    readers_records = {
        i: sorted(records)
        for i, records in readers_records.items()
    }

    data_dirs = sorted(os.listdir(data_dir))
    record_inds = {
        i: sorted(
            [
                j
                for j, r in enumerate(readers_records[i])
                if r.strip('.dat').strip('.ecg') in data_dirs
            ]
        )
        for i in readers_records
    }
    db_per_record = {
        i: [
            j
            for j in range(len(readers_records))
            if (r + '.dat' in readers_records[j] or r + '.ecg' in readers_records[j])
        ][0]
        for i, r in enumerate(data_dirs)
    }
    rhythms = {}
    ind_per_reader = [0, ] * len(readers)
    for i in range(len(data_dirs)):
        db_ind = db_per_record[i]
        reader = readers[db_ind]
        reader_records_pointer = ind_per_reader[db_ind]
        record_ind = record_inds[db_ind][reader_records_pointer]
        record = reader.read_record(record_ind)
        rhythms[i] = record['rhythms']
        ind_per_reader[db_per_record[i]] += 1

    return rhythms


def extract_patients_rhythms_rpeaks(
        data_dir: Union[str, Sequence[str]],
        readers: Sequence[PhysioReader],
) -> Tuple[Dict[int, np.ndarray], List[str]]:
    readers_records = {
        i: reader.reader.records
        for i, reader in enumerate(readers)
    }
    readers_records = {
        i: sorted(records)
        for i, records in readers_records.items()
    }

    if isinstance(data_dir, List) or isinstance(data_dir, Tuple):
        data_dirs = []
        for d in data_dir:
            data_dirs += os.listdir(d)

        data_dirs = sorted(data_dirs)

    else:
        data_dirs = sorted(os.listdir(data_dir))

    record_inds = {
        i: sorted(
            [
                j
                for j, r in enumerate(readers_records[i])
                if (readers[i].db_name + '_' + r.strip('.dat').strip('.ecg')) in data_dirs
            ]
        )
        for i in readers_records
    }
    db_per_record = {
        i: [
            j
            for j in range(len(readers_records))
            if (
                    readers[j].db_name == r.split('_')[0]
                    and
                    (r.split('_')[-1] + '.dat' in readers_records[j] or r.split('_')[-1] + '.ecg' in readers_records[j])
            )
        ][0]
        for i, r in enumerate(data_dirs)
    }
    rhythms = {}
    ind_per_reader = [0, ] * len(readers)
    files_added_by_order = []
    for i in range(len(data_dirs)):
        db_ind = db_per_record[i]
        reader = readers[db_ind]
        reader_records_pointer = ind_per_reader[db_ind]
        record_ind = record_inds[db_ind][reader_records_pointer]
        record = reader.read_record(record_ind)
        rhythms[i] = record['rhythms']
        files_added_by_order.append(reader.reader.files[record_ind])
        ind_per_reader[db_ind] += 1

    return rhythms, files_added_by_order


def extract_patients_unique_rhythms(
        data_dir: Union[str, Sequence[str]],
        readers: Sequence[PhysioReader],
        rhythms_mapping: Dict[int, int] = None,
        rpeaks_data: bool = False,
) -> Tuple[Dict[int, np.ndarray], List[str]]:
    if rpeaks_data:
        rhythms, files_added_by_order = extract_patients_rhythms_rpeaks(
            data_dir=data_dir,
            readers=readers,
        )

    else:
        rhythms = extract_patients_rhythms(
            data_dir=data_dir,
            readers=readers,
        )
        files_added_by_order = None

    for i in rhythms:
        rhythms[i] = np.unique(rhythms[i])

        if rhythms_mapping is not None:
            for r in rhythms_mapping:
                rhythms[i][rhythms[i] == r] = rhythms_mapping[r]

    return rhythms, files_added_by_order


def estimate_alpha_threshold(
        probs: np.ndarray,
        labels: np.ndarray,
        alpha: float,
) -> float:
    thresholds = np.linspace(start=0.001, stop=0.999, num=100)
    min_fpr = 1
    min_fpr_t = thresholds[0]
    for t in thresholds:
        probs_copy = probs.copy()
        probs_copy[probs >= t] = 1
        probs_copy[probs < t] = 0

        cm = sk_metrics.confusion_matrix(probs_copy, labels)
        tn, fn, tp, fp = cm[0][0], cm[1][0], cm[1][1], cm[0][1]
        fpr = fp / (fp + tn)

        if fpr < min_fpr:
            min_fpr = fpr
            min_fpr_t = t

        if fpr <= alpha:
            return t

    else:
        return min_fpr_t


def estimate_beta_threshold(
        probs: np.ndarray,
        labels: np.ndarray,
        beta: float,
) -> float:
    thresholds = np.linspace(start=0.999, stop=0.001, num=100)
    max_tpr = 1
    max_tpr_t = thresholds[0]
    for t in thresholds:
        probs_copy = probs.copy()
        probs_copy[probs >= t] = 1
        probs_copy[probs < t] = 0

        cm = sk_metrics.confusion_matrix(probs_copy, labels)
        tn, fn, tp, fp = cm[0][0], cm[1][0], cm[1][1], cm[0][1]
        tpr = tp / (tp + fn)

        if tpr > max_tpr:
            max_tpr = tpr
            max_tpr_t = t

        if tpr >= beta:
            return t

    else:
        return max_tpr_t


def estimate_threshold(
        probs: np.ndarray,
        labels: np.ndarray,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
) -> float:
    assert (alpha is not None or beta is not None) and not (alpha is not None and beta is not None)

    if alpha is not None:
        return estimate_alpha_threshold(
            probs=probs,
            labels=labels,
            alpha=alpha,
        )

    else:
        return estimate_beta_threshold(
            probs=probs,
            labels=labels,
            beta=beta,
        )


def make_labels(
        rhythms_per_sample: Dict[int, np.ndarray],
        rhythms: Sequence[int],
) -> Sequence[np.ndarray]:
    labels_per_rhythm = [
        np.zeros(len(rhythms_per_sample))
        for _ in range(len(rhythms))
    ]

    for r, rhythm in enumerate(rhythms):
        for i, p in enumerate(rhythms_per_sample):
            if rhythm in rhythms_per_sample[p]:
                labels_per_rhythm[r][i] = 1

    return labels_per_rhythm
