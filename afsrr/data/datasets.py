from torch.utils.data import Dataset, DataLoader
from torch import from_numpy, Tensor, float32, int64
from typing import Dict, List, Union, Sequence, Optional, Tuple
from afsrr.utils.defaults import (
    OTHER_KEY,
    GT_TENSOR_INPUTS_KEY,
    GT_TENSOR_PREDICITONS_KEY,
)

import os
import h5py
import torch
import numpy as np


class RPeaksDataset(Dataset):
    """
    Dataset class for handling R-peak data from ECG recordings.
    
    This dataset loads and processes data files, optionally filtering for normal sinus rhythm (NSR)
    segments and handling different data splitting strategies.

    Args:
        mode (str): Dataset mode ('Train', 'Val', or 'Test')
        temporal_horizon (int): Number of time steps to look back
        prediction_horizon (int, optional): Number of time steps to predict ahead. Defaults to 1
        peaks_per_sample (int, optional): Number of R-peaks per sample. Defaults to 50
        peaks_step (int, optional): Step size between consecutive peak windows. Defaults to 5
        dir_path (Union[str, Sequence[str]], optional): Path(s) to data directory. Defaults to None
        nsr_only (bool, optional): Whether to only use normal sinus rhythm segments. Defaults to False
        record_length_to_use (Optional[int], optional): Fixed length to use from each record. Defaults to None
        with_labels (bool, optional): Whether to include rhythm labels. Defaults to False
        invalid_inds (Optional[List[int]], optional): Indices of records to exclude. Defaults to []
        records_paths (Optional[Sequence[str]], optional): Direct paths to record files. Defaults to None
        nsr_from_start (bool, optional): Take NSR segment from start of recording. Defaults to True
        nsr_from_end (bool, optional): Take NSR segment from end of recording. Defaults to False
        nsr_from_middle (bool, optional): Take NSR segment from middle of recording. Defaults to False
        merge_modes (bool, optional): Whether to merge train/val/test data. Defaults to False
    """

    def __init__(
            self,
            mode: str,
            temporal_horizon: int,
            prediction_horizon: int = 1,
            peaks_per_sample: int = 50,
            peaks_step: int = 5,
            dir_path: Union[str, Sequence[str]] = None,
            nsr_only: bool = False,
            record_length_to_use: Optional[int] = None,
            with_labels: bool = False,
            invalid_inds: Optional[List[int]] = [],
            records_paths: Optional[Sequence[str]] = None,
            nsr_from_start: bool = True,
            nsr_from_end: bool = False,
            nsr_from_middle: bool = False,
            merge_modes: bool = False,
    ):
        super().__init__()

        assert not (dir_path is None and records_paths is None)

        if invalid_inds is None:
            invalid_inds = []

        self._mode = mode
        self._temporal_horizon = temporal_horizon
        self._prediction_horizon = prediction_horizon
        self._record_length_to_use = record_length_to_use
        self._nsr_only = nsr_only
        self._peaks_per_sample = peaks_per_sample
        self._peaks_step = peaks_step
        self._with_labels = with_labels
        self._invalid_inds = invalid_inds
        self._records_paths = records_paths
        self._nsr_from_start = nsr_from_start
        self._nsr_from_end = nsr_from_end
        self._nsr_from_middle = nsr_from_middle
        self._merge_modes = merge_modes

        assert not (
                (nsr_from_start and nsr_from_middle) or
                (nsr_from_start and nsr_from_end) or
                (nsr_from_middle and nsr_from_end)
        )

        if records_paths is not None:
            self._patients_dirs = records_paths

        else:
            patients_dirs = os.listdir(dir_path)
            self._patients_dirs = [
                    os.path.join(dir_path, d)
                    for d in patients_dirs
                    if os.path.isdir(os.path.join(dir_path, d))
            ]

        self._patients_dirs = sorted(self._patients_dirs)
        self.files = [
            os.path.join(d, mode, d.split(os.sep)[-1].split('_')[-1] + '.h5')
            for d in self._patients_dirs
        ]
        self.n_patients = len(self.files)

        self._lengths = []
        self._effective_lengths = []
        xs = []
        ys = []
        invalid_files = []
        for f_ind, f in enumerate(self.files):
            try:
                if merge_modes:
                    f_train = os.path.join(
                        os.sep.join(f.split(os.sep)[:-2]),
                        'Train',
                        self._patients_dirs[f_ind].split(os.sep)[-1].split('_')[-1] + '.h5',
                    )
                    f_val = os.path.join(
                        os.sep.join(f.split(os.sep)[:-2]),
                        'Val',
                        self._patients_dirs[f_ind].split(os.sep)[-1].split('_')[-1] + '.h5',
                    )
                    f_test = os.path.join(
                        os.sep.join(f.split(os.sep)[:-2]),
                        'Test',
                        self._patients_dirs[f_ind].split(os.sep)[-1].split('_')[-1] + '.h5',
                    )
                    with h5py.File(f_train, 'r') as h5file:
                        data_train = h5file['record']
                        x_train = data_train['x'][:]
                        y_train = data_train['y'][:]

                    with h5py.File(f_val, 'r') as h5file:
                        data_val = h5file['record']
                        x_val = data_val['x'][:]
                        y_val = data_val['y'][:]

                    with h5py.File(f_test, 'r') as h5file:
                        data_test = h5file['record']
                        x_test = data_test['x'][:]
                        y_test = data_test['y'][:]

                    x = np.concatenate([x_train, x_val, x_test], axis=-1)
                    y = np.concatenate([y_train, y_val, y_test], axis=-1)

                else:
                    with h5py.File(f, 'r') as h5file:
                        data = h5file['record']
                        x = data['x'][:]
                        y = data['y'][:]

                if nsr_only:
                    # Find all NSR indices
                    nsr_indices = np.concatenate([np.where(y == 0)[0], np.where(y == 1)[0]])

                    if not len(nsr_indices) >= record_length_to_use:
                        invalid_files.append(f_ind)

                    # Find the longest continuous NSR sequence
                    start_nsr, end_nsr = self._find_longest_nsr_sequence(nsr_indices)

                    # Take the longest sequence as the data sequence
                    if len(nsr_indices) == 0:
                        invalid_files.append(f_ind)
                        nsr_x = x
                        nsr_y = y

                    else:
                        nsr_x = x[nsr_indices[start_nsr]:nsr_indices[end_nsr]]
                        nsr_y = y[nsr_indices[start_nsr]:nsr_indices[end_nsr]]

                    if not len(nsr_x) >= record_length_to_use:
                        invalid_files.append(f_ind)

                    if not (np.sum(np.abs(nsr_y - 1)) == 0 or np.sum(np.abs(nsr_y)) == 0):
                        invalid_files.append(f_ind)

                    if nsr_from_start:
                        xs.append(nsr_x[:record_length_to_use][None, None, ...])
                        ys.append(nsr_y[:record_length_to_use][None, None, ...])

                    elif nsr_from_middle:
                        start_i = (len(nsr_x) - record_length_to_use) // 2
                        xs.append(nsr_x[start_i:(start_i + record_length_to_use)][None, None, ...])
                        ys.append(nsr_y[start_i:(start_i + record_length_to_use)][None, None, ...])

                    elif nsr_from_end:
                        xs.append(nsr_x[-record_length_to_use:][None, None, ...])
                        ys.append(nsr_y[-record_length_to_use:][None, None, ...])

                    self._effective_lengths.append(record_length_to_use)

                else:
                    if record_length_to_use is None:
                        x_rec = x[None, None, ...]
                        y_rec = y[None, None, ...]

                    else:
                        x_rec = x[None, None, :record_length_to_use]
                        y_rec = y[None, None, :record_length_to_use]

                    effective_length = (
                        (
                            x_rec.shape[-1] -
                            ((temporal_horizon + prediction_horizon) * peaks_per_sample)
                        ) // peaks_step
                    )

                    if effective_length <= 0:
                        invalid_files.append(f_ind)

                    xs.append(x_rec)
                    ys.append(y_rec)
                    self._effective_lengths.append(effective_length)

            except:
                print(f"Skipping {f} -- Not Found")
                invalid_files.append(f_ind)

        self.invalid_files = np.unique(invalid_files + invalid_inds).tolist()
        print(
            f"Detected {len(self.invalid_files)} / {len(self.files)} invalid files"
            f" for 'record_length_to_use'={record_length_to_use}"
        )

        self.files = [
            f
            for f_ind, f in enumerate(self.files)
            if (f_ind not in self.invalid_files)
        ]
        self._af_labels = np.array(
            [
                1
                if 'af' in f.lower() else
                0
                for f in self.files
            ]
        )
        self.n_patients = len(self.files)
        effective_lengths = [
            np.array([l, ])
            for l_ind, l in enumerate(self._effective_lengths)
            if (l_ind not in self.invalid_files)
        ]
        xs = [
            x
            for x_ind, x in enumerate(xs)
            if (x_ind not in self.invalid_files)
        ]
        ys = [
            y
            for y_ind, y in enumerate(ys)
            if (y_ind not in self.invalid_files)
        ]
        self._effective_lengths = np.concatenate(effective_lengths, axis=0)

        if record_length_to_use is not None:
            self._length = (
                (
                    record_length_to_use -
                    ((temporal_horizon + prediction_horizon) * peaks_per_sample)
                ) // peaks_step
            )
            ys = [y for y in ys if y.shape[-1] == record_length_to_use]
            xs = [x for x in xs if x.shape[-1] == record_length_to_use]
            self._xs = np.concatenate(xs, axis=0)
            self._ys = np.concatenate(ys, axis=0)

        else:
            self._length = max(self._effective_lengths)
            self._xs = xs
            self._ys = ys

    def _find_longest_nsr_sequence(self, nsr_indices: np.ndarray) -> Tuple[int, int]:
        """
        Find the longest continuous sequence of normal sinus rhythm (NSR) beats.

        Args:
            nsr_indices (np.ndarray): Array of indices where NSR was detected

        Returns:
            Tuple[int, int]: Start and end indices of the longest NSR sequence
        """
        # Find all continuous sequences
        nsr_indices_diff = np.diff(nsr_indices)
        longest_sequence_start = 0
        longest_sequence_end = 1
        longest_sequence_length = 1
        current_sequence_start = 0
        current_sequence_end = 1
        current_sequence_length = 1
        for i, diff in enumerate(nsr_indices_diff):
            if diff == 1:
                current_sequence_end += 1
                current_sequence_length += 1

            else:
                if current_sequence_length > longest_sequence_length:
                    longest_sequence_start = current_sequence_start
                    longest_sequence_end = current_sequence_end
                    longest_sequence_length = current_sequence_length

                current_sequence_start = i + 1
                current_sequence_end = i + 2
                current_sequence_length = 1

        if current_sequence_length > longest_sequence_length:
            longest_sequence_start = current_sequence_start
            longest_sequence_end = current_sequence_end

        return longest_sequence_start, longest_sequence_end - 1

    def __enter__(self):
        """Context manager enter method"""
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit method"""
        pass

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: Number of samples in the dataset
        """
        return self._length

    def __getitem__(self, index: int) -> Dict[str, Union[Tensor, Dict[str, Union[np.ndarray, Tensor]]]]:
        """
        Get a single sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve

        Returns:
            Dict containing:
                - GT_TENSOR_INPUTS_KEY: Input tensor of shape (batch, channels, time)
                - GT_TENSOR_PREDICITONS_KEY: Target tensor of shape (batch, channels, time)
                - OTHER_KEY: Dict containing additional data:
                    - x_labels: Input rhythm labels
                    - labels: Target rhythm labels
                    - af_labels: Atrial fibrillation labels
        """
        x_labels = np.zeros((len(self._ys), 20))
        y_labels = np.zeros((len(self._ys), 20))
        if not self._nsr_only:
            xs = [
                np.concatenate(
                    [
                        x[...,
                            ((index % self._effective_lengths[i]) + h) * self._peaks_step:
                            (((index % self._effective_lengths[i]) + h) * self._peaks_step) + self._peaks_per_sample
                        ]
                        for h in range(self._temporal_horizon)
                    ],
                    axis=1,
                )
                for i, x in enumerate(self._xs)
            ]
            x = np.concatenate(xs, 0)

            ys = [
                np.concatenate(
                    [
                        x[...,
                            ((index % self._effective_lengths[i]) + h + t) * self._peaks_step:
                            (((index % self._effective_lengths[i]) + h + t) * self._peaks_step) + self._peaks_per_sample
                        ]
                        for h in range(1, self._prediction_horizon + 1)
                        for t in range(self._temporal_horizon)
                    ],
                    axis=1,
                )
                for i, x in enumerate(self._xs)
            ]
            y = np.concatenate(ys, 0)

        else:
            x = np.concatenate(
                [
                    self._xs[...,
                        ((index + h) * self._peaks_step):((index + h) * self._peaks_step + self._peaks_per_sample)
                    ]
                    for h in range(self._temporal_horizon)
                ],
                axis=1,
            )
            y = np.concatenate(
                [
                    self._xs[...,
                        ((index + t + h) * self._peaks_step):((index + t + h) * self._peaks_step + self._peaks_per_sample)
                    ]
                    for h in range(1, self._prediction_horizon + 1)
                    for t in range(self._temporal_horizon)
                ],
                axis=1,
            )

        if self._with_labels:
            x_l = [
                np.unique(
                    labels[
                        ...,
                        (index * self._peaks_step) % self._effective_lengths[i]:
                        (
                            (index + ((self._temporal_horizon - 1) * self._peaks_step)) %
                            self._effective_lengths[i]
                        ) + self._peaks_per_sample
                    ]
                ).astype(int)[None, :]
                for i, labels in enumerate(self._ys)
            ]
            for l_index, l in enumerate(x_l):
                x_labels[l_index, l[0, :]] = 1

            y_l = [
                np.unique(
                    labels[
                        ...,
                        (
                            (index + (self._temporal_horizon * self._peaks_step)) %
                            self._effective_lengths[i]
                        ) + self._peaks_per_sample:
                        (
                            (index + ((self._temporal_horizon + 1) * self._peaks_step)) %
                            self._effective_lengths[i]
                        ) + self._peaks_per_sample
                    ]
                ).astype(int)[None, :]
                for i, labels in enumerate(self._ys)
            ]
            for l_index, l in enumerate(y_l):
                y_labels[l_index, l[0, :]] = 1

        x = from_numpy(x).type(torch.float32)
        y = from_numpy(y).type(torch.float32)
        af_labels = from_numpy(self._af_labels).type(torch.float32)

        sample = {
            GT_TENSOR_INPUTS_KEY: x,
            GT_TENSOR_PREDICITONS_KEY: y,
            OTHER_KEY: {
                'x_labels': x_labels,
                'labels': y_labels,
                'af_labels': af_labels
            },
        }

        return sample
