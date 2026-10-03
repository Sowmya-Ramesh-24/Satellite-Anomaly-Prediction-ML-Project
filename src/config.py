"""Central configuration: paths, modelling choices, features."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW, PROCESSED, RESULTS = ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "results"
ANOMALY_FILE = RAW / "anom5j.xls"
SATCAT_FILE = RAW / "satcat.csv"
SUNSPOT_FILE = RAW / "sunspot.csv"

# --- data cleaning
SUNSPOT_START = "1976-01-01"        # anomalies before this have no sunspot data
MIN_TTE_OBSERVATIONS = 20           # keep satellites with more than 20 valid TTE rows
EXCLUDE_PREFIXES = ("STS-",)        # space shuttle missions are not long-lived satellites
EXCLUDE_SATELLITES = ("SCATHA", "ECS 1")
ORBIT_CLASSES = ["G", "I", "C", "E", "P", "V"]   # everything else -> reference level
CENSOR_MODE = "none"                # retained for compatibility with older notebooks

# --- evaluation
N_SPLITS = 10                       # shuffled KFold for the Stanford experiment
RANDOM_STATE = 42

FEATURE_GROUPS = {
    "space": ["month_sin", "month_cos", "sun_27", "sun_81", "sun_trend"],
    "satellite": [f"orbit_{c}" for c in ORBIT_CLASSES],
    "history": ["prev_log_gap", "log_n_prev"],
    "satcat": ["age_years", "inclination", "perigee", "apogee"],
}
MODEL_FEATURE_GROUPS = ["space", "satellite", "history"]   # used by all three model notebooks
