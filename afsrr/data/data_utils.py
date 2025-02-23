from afsrr import RAW_DATA_DIR
from multiprocessing import Pool
from typing import Sequence, List, Optional
from afsrr.data.physionet_readers import PhysioReader
from afsrr.data.physionet_writers import PhysioRecorder
from afsrr.data.physionet_processors import PhysioProcessor

import os
import h5py as h5
import numpy as np


def process_single_file(args):
    file_path, final_dir, train_ratio, val_ratio, frequency = args
    print(f"Processing file: {file_path}")
    
    train_file_path = os.path.join(final_dir, 'Train', file_path.split(os.sep)[-1])
    val_file_path = os.path.join(final_dir, 'Val', file_path.split(os.sep)[-1])
    test_file_path = os.path.join(final_dir, 'Test', file_path.split(os.sep)[-1])

    if (
            os.path.isfile(train_file_path) and
            os.path.isfile(val_file_path) and
            os.path.isfile(test_file_path)
    ):
        return

    with h5.File(file_path, 'r') as file:
        record = file['record']
        x = np.diff((record['x'][:] / frequency))
        y = record['y'][:]
        qrs = record['qrs'][:]

        assert np.isnan(x).sum() == 0
        assert np.isnan(y).sum() == 0
        assert np.isnan(qrs).sum() == 0

        assert np.isinf(x).sum() == 0
        assert np.isinf(y).sum() == 0
        assert np.isinf(qrs).sum() == 0

        n = x.shape[0]
        assert (n == y.shape[0] - 1) or (y.shape[0] == 0)
        assert n == qrs.shape[0] - 1

        train_end = int(train_ratio * n)
        val_end = train_end + int(val_ratio * n)

        train_x = x[:train_end]
        val_x = x[train_end:val_end]
        test_x = x[val_end:]

        train_y = y[:train_end]
        val_y = y[train_end:val_end]
        test_y = y[val_end:]

        train_qrs = qrs[:train_end]
        val_qrs = qrs[train_end:val_end]
        test_qrs = qrs[val_end:]

    print(f"# Samples in file: {train_x.shape[0]}")
    os.makedirs(os.path.join(final_dir), exist_ok=True)
    os.makedirs(os.path.join(final_dir, 'Train'), exist_ok=True)
    os.makedirs(os.path.join(final_dir, 'Val'), exist_ok=True)
    os.makedirs(os.path.join(final_dir, 'Test'), exist_ok=True)

    with h5.File(train_file_path, 'w') as file:
        dataset = file.create_group('/record')
        dataset.create_dataset('x', shape=train_x.shape, dtype=train_x.dtype, data=train_x)
        dataset.create_dataset('y', shape=train_y.shape, dtype=train_y.dtype, data=train_y)
        dataset.create_dataset('qrs', shape=train_qrs.shape, dtype=train_qrs.dtype, data=train_qrs)

    with h5.File(val_file_path, 'w') as file:
        dataset = file.create_group('/record')
        dataset.create_dataset('x', shape=val_x.shape, dtype=val_x.dtype, data=val_x)
        dataset.create_dataset('y', shape=val_y.shape, dtype=val_y.dtype, data=val_y)
        dataset.create_dataset('qrs', shape=val_qrs.shape, dtype=val_qrs.dtype, data=val_qrs)

    with h5.File(test_file_path, 'w') as file:
        dataset = file.create_group('/record')
        dataset.create_dataset('x', shape=test_x.shape, dtype=test_x.dtype, data=test_x)
        dataset.create_dataset('y', shape=test_y.shape, dtype=test_y.dtype, data=test_y)
        dataset.create_dataset('qrs', shape=test_qrs.shape, dtype=test_qrs.dtype, data=test_qrs)


def split_data(
        raw_dir: str,
        save_dir: str,
        db_name: str,
        frequency: float,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        n_workers: int = 1,
):
    dirs = sorted(os.listdir(raw_dir))
    files = [
        os.path.join(raw_dir, d, d + '.h5')
        for d in dirs
        if os.path.isfile(os.path.join(raw_dir, d, d + '.h5'))
    ]
    final_dirs = [
        os.path.join(save_dir, db_name + '_' + f.split(os.sep)[-2])
        for f in files
    ]

    args = [
        (f, fd, train_ratio, val_ratio, frequency)
        for f, fd in zip(files, final_dirs)
    ]

    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            pool.map(process_single_file, args)
    else:
        for arg in args:
            process_single_file(arg)


def get_train_val_test_records_from_lines(lines: Sequence[str]) -> List[str]:
    ltafdb_lines = lines[0].strip(os.linesep).split(': ')
    if len(ltafdb_lines) > 1:
        ltafdb_records = [
            os.path.join(RAW_DATA_DIR, "ltafdb", f"{rec}.h5")
            for rec in ltafdb_lines[-1].split(' ')
        ]

    else:
        ltafdb_records = []

    afdb_lines = lines[0].strip(os.linesep).split(': ')
    if len(afdb_lines) > 1:
        afdb_records = [
            os.path.join(RAW_DATA_DIR, "afdb", f"{rec}.h5")
            for rec in afdb_lines[-1].split(' ')
        ]

    else:
        afdb_records = []

    nsrdbrr_lines = lines[0].strip(os.linesep).split(': ')
    if len(nsrdbrr_lines) > 1:
        nsrdbrr_records = [
            os.path.join(RAW_DATA_DIR, "nsrdbrr", f"{rec}.h5")
            for rec in nsrdbrr_lines[-1].split(' ')
        ]

    else:
        nsrdbrr_records = []

    thew_lines = lines[0].strip(os.linesep).split(': ')
    if len(thew_lines) > 1:
        thew_records = [
            os.path.join(RAW_DATA_DIR, "thew", f"{rec}.h5")
            for rec in thew_lines[-1].split(' ')
        ]

    else:
        thew_records = []

    records = ltafdb_records + afdb_records + nsrdbrr_records + thew_records

    return records


def get_train_val_test_split():
    split_config = os.path.join(
        RAW_DATA_DIR,
        "arrhythmia_classification_train_val_test_split.txt",
    )
    with open(split_config, 'r') as f:
        split = f.readlines()

    training_lines = split[1:5]
    validation_lines = split[7:11]
    testing_lines = split[13:17]

    train_records = get_train_val_test_records_from_lines(training_lines)
    val_records = get_train_val_test_records_from_lines(validation_lines)
    test_records = get_train_val_test_records_from_lines(testing_lines)

    return train_records, val_records, test_records


def write_record(index, rdr, rec, prc, save_dir):
    try:
        record = rdr.read_record(index)

        current_save_dir = os.path.join(save_dir, record['file_name'])
        os.makedirs(
            current_save_dir,
            exist_ok=True,
        )

        if os.path.isfile(os.path.join(current_save_dir, record['file_name'] + '.h5')):
            return

        if prc is not None:
            processed_record = prc.process_record(
                record=record['signal'][:, 0],
                qrs=record['qrs'],
                labels=record['rhythms'],
                frequency=rdr.frequency,
            )
            ecg = processed_record['signal']
            labels = processed_record['rhythms']
            qrs = processed_record['qrs']

        else:
            ecg = record['signal']
            labels = record['rhythms']
            qrs = record['qrs']

        rec.convert_single_record(
            x=qrs,
            raw_file_name=record['file_name'],
            y=labels,
            record_id=index,
            raw_qrs_inds=qrs,
            additional_arrays={'ecg': ecg},
            save_dir=current_save_dir,
        )

    except:
        print(f"\n ------------- Failed on record {index} ------------- \n")


def parallel_processing(
        n_workers: int,
        inds: Sequence[int],
        rdr: PhysioReader,
        rec: PhysioRecorder,
        prc: Optional[PhysioProcessor],
        save_dir: str,
):
    args = list(
        zip(
            inds,
            [rdr, ] * len(inds),
            [rec, ] * len(inds),
            [prc, ] * len(inds),
            [save_dir, ] * len(inds),
        )
    )

    with Pool(processes=n_workers) as pool:
        pool.starmap(write_record, args)
