from afsrr.data.data_utils import split_data

from afsrr.data.data_utils import parallel_processing
from afsrr.data.physionet_readers import PhysioReader
from afsrr.data.physionet_writers import PhysioRecorder
from afsrr.data.physionet_processors import PhysioProcessor
from afsrr import PROCESSED_DATA_DIR, RAW_THEW_DB, RAW_AFDB, RAW_LTAFDB, RAW_NSRDBRR

import os


train_ratio = 0.6
val_ratio = 0.2
n_workers = 8
readers = (
    PhysioReader(
        db_path=RAW_LTAFDB,
        db_name='ltafdb',
    ),
    PhysioReader(
        db_path=RAW_AFDB,
        db_name='afdb',
    ),
    PhysioReader(
        db_path=RAW_NSRDBRR,
        db_name='nsrdbrr',
    ),
    PhysioReader(
        db_path=RAW_THEW_DB,
        db_name='thew',
    ),
)
processed_train_dir = os.path.join(PROCESSED_DATA_DIR, 'Train')
os.makedirs(processed_train_dir, exist_ok=True)
processed_val_dir = os.path.join(PROCESSED_DATA_DIR, 'Val')
os.makedirs(processed_val_dir, exist_ok=True)
processed_test_dir = os.path.join(PROCESSED_DATA_DIR, 'Test')
os.makedirs(processed_test_dir, exist_ok=True)
processors = (
    PhysioProcessor(
        output_signal='RR',
        trim_n_seconds=500,
    ),
    PhysioProcessor(
        output_signal='RR',
    ),
    PhysioProcessor(
        output_signal='RR',
    ),
    PhysioProcessor(
        output_signal='RR',
    ),
)
if __name__ == "__main__":
    for reader, processor in zip(readers, processors):
        print(f"Processing the raw {reader.db_name}")
        indices = list(range(len(reader)))
        processed_save_dir = os.path.join(
            PROCESSED_DATA_DIR,
            reader.db_name
        )
        os.makedirs(processed_save_dir, exist_ok=True)

        recorder = PhysioRecorder(
            save_dir=processed_save_dir,
            db_name=reader.db_name,
        )
        parallel_processing(
            n_workers=n_workers,
            inds=indices,
            rdr=reader,
            rec=recorder,
            prc=processor,
            save_dir=processed_save_dir,
        )

    for reader in readers:
        print(f"Generating the processed train, validation, and test sets for {reader.db_name}")
        raw_dir = os.path.join(
            PROCESSED_DATA_DIR,
            reader.db_name
        )
        split_data(
            raw_dir=raw_dir,
            save_dir=processed_train_dir,
            db_name=reader.db_name,
            frequency=reader.frequency,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            n_workers=n_workers,
        )
