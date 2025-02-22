from pkg_resources import get_distribution, DistributionNotFound
from pathlib import Path

import os

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version

except DistributionNotFound:
    __version__ = 'unknown'

finally:
    del get_distribution, DistributionNotFound

PROJECT_ROOT = Path(Path(__file__).resolve().parents[1])
if str(PROJECT_ROOT).startswith(os.getcwd()):
    PROJECT_ROOT = PROJECT_ROOT.relative_to(os.getcwd())

LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

ANALYSIS_LOGS_DIR = os.path.join(LOGS_DIR, 'analysis')
os.makedirs(ANALYSIS_LOGS_DIR, exist_ok=True)

EXPERIMENTS_LOGS_DIR = os.path.join(LOGS_DIR, 'experiments')
os.makedirs(EXPERIMENTS_LOGS_DIR, exist_ok=True)

DEMO_LOGS_DIR = os.path.join(LOGS_DIR, 'demo')
os.makedirs(DEMO_LOGS_DIR, exist_ok=True)

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

DEMO_DATA_DIR = os.path.join(DATA_DIR, 'demo')
os.makedirs(DEMO_DATA_DIR, exist_ok=True)

RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
os.makedirs(RAW_DATA_DIR, exist_ok=True)

RAW_LTAFDB = os.path.join(RAW_DATA_DIR, "ltafdb")
os.makedirs(RAW_LTAFDB, exist_ok=True)

RAW_AFDB = os.path.join(RAW_DATA_DIR, "afdb")
os.makedirs(RAW_AFDB, exist_ok=True)

RAW_NSRDBRR = os.path.join(RAW_DATA_DIR, "nsrdbrr")
os.makedirs(RAW_NSRDBRR, exist_ok=True)

RAW_THEW_DB = os.path.join(RAW_DATA_DIR, "thew")
os.makedirs(RAW_THEW_DB, exist_ok=True)

PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

