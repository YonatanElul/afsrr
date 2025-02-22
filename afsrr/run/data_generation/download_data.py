from afsrr import RAW_AFDB, RAW_LTAFDB, RAW_NSRDBRR

import os


if __name__ == "__main__":
    os.system(f"aws s3 sync --no-sign-request s3://physionet-open/afdb/1.0.0/ {RAW_AFDB}")
    os.system(f"aws s3 sync --no-sign-request s3://physionet-open/ltafdb/1.0.0/ {RAW_LTAFDB}")
    os.system(f"aws s3 sync --no-sign-request s3://physionet-open/nsr2db/1.0.0/ {RAW_NSRDBRR}")

