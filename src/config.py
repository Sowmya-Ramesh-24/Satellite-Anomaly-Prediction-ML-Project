"""Central configuration: paths, modelling choices, features."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW, PROCESSED, RESULTS = ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "results"
ANOMALY_FILE = RAW / "anom5j.xls"
SATCAT_FILE = RAW / "satcat.csv"
SUNSPOT_FILE = RAW / "sunspot.csv"

# --- data cleaning
SUNSPOT_START = "1976-01-01"        # anomalies before this have no sunspot data
MIN_EVENTS = 10                     # keep satellites with at least this many anomaly-days
EXCLUDE_PREFIXES = ("STS-",)        # space shuttle missions are not long-lived satellites
ORBIT_CLASSES = ["G", "I", "C", "E", "P", "V"]   # everything else -> reference level
CENSOR_MODE = "satcat"              # "none" | "satcat" | "data_end"  (see create_tte.py)

# --- prediction task
HORIZON_DAYS = 7                    # label = 1 if the NEXT anomaly happens within this many days

# --- evaluation
N_SPLITS = 5                        # GroupKFold by satellite
THRESHOLD = 0.5                     # probability cut-off for precision/recall/F1/confusion matrix
RANDOM_STATE = 42

FEATURE_GROUPS = {
    "space": ["month_sin", "month_cos", "sun_27", "sun_81", "sun_trend"],
    "satellite": ["log_alt"] + [f"orbit_{c}" for c in ORBIT_CLASSES],
    "history": ["prev_log_gap", "log_n_prev"],
    "satcat": ["age_years", "inclination", "perigee", "apogee"],
}
MODEL_FEATURE_GROUPS = ["space", "satellite", "history"]   # used by all three model notebooks
