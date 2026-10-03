"""Clean anomaly records, build one-row-per-satellite attributes, join satcat."""
import re
import numpy as np
import pandas as pd
from src.config import ORBIT_CLASSES


def norm_name(x) -> str:
    """'GOES-5' and 'GOES 5' -> 'GOES5'."""
    return re.sub(r"[^A-Z0-9]", "", str(x).upper())


def clean_anomalies(raw: pd.DataFrame, start, exclude_prefixes) -> pd.DataFrame:
    """One row per (satellite, day). Several reports on the same day are collapsed,
    otherwise we would create zero-day gaps. ACOMMENT is deliberately NOT kept:
    it contains space-weather conditions at the time of the anomaly (leakage)."""
    df = raw[["BIRD", "ADATE", "ORBIT", "ALT"]].copy()
    df["BIRD"] = df.BIRD.astype(str).str.strip().str.upper()
    df["ORBIT"] = df.ORBIT.astype(str).str.strip().str.upper()
    df["ADATE"] = pd.to_datetime(df.ADATE, errors="coerce")
    df["ALT"] = df.ALT.where(df.ALT > 0)           # 0 = missing
    df = df.dropna(subset=["ADATE"])
    df = df[df.ADATE >= pd.Timestamp(start)]
    df = df[~df.BIRD.str.startswith(tuple(exclude_prefixes))]
    return df


def satellite_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-satellite attributes taken from the anomaly file itself (available for ALL satellites)."""
    g = df.groupby("BIRD")
    sat = pd.DataFrame({
        "orbit": g.ORBIT.agg(lambda s: s.mode().iat[0]),
        "alt_km": g.ALT.median(),
        "first_anomaly": g.ADATE.min(),
    })
    sat["log_alt"] = np.log1p(sat.alt_km)
    for c in ORBIT_CLASSES:
        sat[f"orbit_{c}"] = (sat.orbit == c).astype(int)
    return sat


def match_satcat(sats: pd.DataFrame, satcat: pd.DataFrame) -> pd.DataFrame:
    """Attach satcat info by normalised name. Only unambiguous payload matches whose
    launch date precedes the first recorded anomaly are accepted."""
    pay = satcat[satcat.OBJECT_TYPE == "PAY"].copy()
    pay["key"] = pay.OBJECT_NAME.map(norm_name)
    pay = pay[pay.key.map(pay.key.value_counts()) == 1].set_index("key")
    pay = pay.rename(columns=str.lower)[["launch_date", "decay_date", "inclination", "perigee", "apogee"]]
    out = sats.copy()
    out["key"] = out.index.map(norm_name)
    out = out.join(pay, on="key").drop(columns="key")
    ok = out.launch_date.notna() & (out.launch_date <= out.first_anomaly)
    out.loc[~ok, ["launch_date", "decay_date", "inclination", "perigee", "apogee"]] = np.nan
    out["in_satcat"] = ok
    return out
