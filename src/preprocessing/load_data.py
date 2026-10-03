"""Read the three raw files. No cleaning here."""
import pandas as pd
from src.config import ANOMALY_FILE, SATCAT_FILE, SUNSPOT_FILE


def load_anomalies(path=ANOMALY_FILE) -> pd.DataFrame:
    return pd.read_excel(path)


def load_satcat(path=SATCAT_FILE) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["LAUNCH_DATE", "DECAY_DATE"])


def load_sunspot(path=SUNSPOT_FILE) -> pd.Series:
    """Daily sunspot number indexed by date (the file is not stored in date order)."""
    df = pd.read_csv(path)
    idx = pd.to_datetime(dict(year=df.Year, month=df.Month, day=df.Day))
    return pd.Series(df.SSN.values, index=idx, name="ssn").sort_index()
