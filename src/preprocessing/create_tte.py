"""Build the continuous time-to-event table used by the regression experiment."""
import numpy as np
import pandas as pd


def build_tte_table(events, sats, sunspot, min_tte_observations=20, **_ignored) -> pd.DataFrame:
    """Return valid consecutive-anomaly gaps with continuous TTE in days.

    Only gaps to a known next anomaly are retained.  Features describe the
    satellite and information available at the start of each gap.
    """
    ss = pd.DataFrame({"sun_27": sunspot.rolling(27, min_periods=1).mean(),
                       "sun_81": sunspot.rolling(81, min_periods=1).mean()})
    ss["sun_trend"] = ss.sun_27 - ss.sun_81

    rows = []
    for sat, g in events.groupby("BIRD"):
        dates = [pd.Timestamp(d) for d in sorted(g.ADATE.unique())]
        if sat in {"SCATHA", "ECS 1"}:
            continue
        s = sats.loc[sat]
        valid_gaps = [
            (dates[i + 1] - dates[i]).days
            for i in range(len(dates) - 1)
            if 0 < (dates[i + 1] - dates[i]).days < 365
        ]
        if len(valid_gaps) <= min_tte_observations:
            continue
        for i, start in enumerate(dates[:-1]):
            tte = (dates[i + 1] - start).days
            if not 0 < tte < 365:
                continue
            rows.append(dict(
                sat=sat, start=start, tte=tte, orbit=s.orbit, in_satcat=s.in_satcat,
                month_sin=np.sin(2 * np.pi * start.month / 12),
                month_cos=np.cos(2 * np.pi * start.month / 12),
                **ss.loc[start].to_dict(),
                prev_log_gap=np.log((start - dates[i - 1]).days) if i > 0 else np.nan,
                log_n_prev=np.log1p(i),
                **{c: s[c] for c in s.index if c.startswith("orbit_")},
                age_years=(start - s.launch_date).days / 365.25 if pd.notna(s.launch_date) else np.nan,
                inclination=s.inclination, perigee=s.perigee, apogee=s.apogee,
            ))
    return pd.DataFrame(rows)
