from afsrr import RAW_AFDB, RAW_LTAFDB, RAW_NSRDBRR

import os


if __name__ == "__main__":
    os.system(f'wget -r -N -c -np -nH --cut-dirs=3 -P {RAW_AFDB} -A "*.dat,*.hea,*.atr,*.ecg,*.qrs" https://physionet.org/files/afdb/1.0.0/')
    os.system(f'wget -r -N -c -np -nH --cut-dirs=3 -P {RAW_LTAFDB} -A "*.dat,*.hea,*.atr,*.ecg,*.qrs" https://physionet.org/files/ltafdb/1.0.0/')
    os.system(f'wget -r -N -c -np -nH --cut-dirs=3 -P {RAW_NSRDBRR} -A "*.dat,*.hea,*.atr,*.ecg,*.qrs" https://physionet.org/files/nsr2db/1.0.0/')

