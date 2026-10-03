"""Turn anomaly dates into a time-to-event table (one row per gap, with censoring)."""
import numpy as np
import pandas as pd


def _censor_end(sat_row, mode, data_end):
    """Date until which a satellite is known to have been observed anomaly-free."""
    if mode == "none":
        return None
    if mode == "data_end":
        return data_end
    # "satcat": only satellites matched in satcat; stop at decay date or end of the database
    if not sat_row.in_satcat:
        return None
    return min(sat_row.decay_date, data_end) if pd.notna(sat_row.decay_date) else data_end


def build_tte_table(events, sats, sunspot, min_events, censor_mode, data_end) -> pd.DataFrame:
    """Row = gap that starts at an anomaly and ends at the next one (event=1) or at the
    end of observation (event=0, right-censored). All features use information
    available at the gap START only, so nothing leaks from the future."""
    ss = pd.DataFrame({"sun_27": sunspot.rolling(27, min_periods=1).mean(),
                       "sun_81": sunspot.rolling(81, min_periods=1).mean()})
    ss["sun_trend"] = ss.sun_27 - ss.sun_81

    rows = []
    for sat, g in events.groupby("BIRD"):
        dates = [pd.Timestamp(d) for d in sorted(g.ADATE.unique())]
        if len(dates) < min_events:
            continue
        s = sats.loc[sat]
        for i, start in enumerate(dates):
            if i + 1 < len(dates):
                dur, ev = (dates[i + 1] - start).days, 1
            else:
                end = _censor_end(s, censor_mode, data_end)
                if end is None or (end - start).days < 1:
                    continue
                dur, ev = (end - start).days, 0
            rows.append(dict(
                sat=sat, start=start, duration=dur, event=ev, orbit=s.orbit, in_satcat=s.in_satcat,
                month_sin=np.sin(2 * np.pi * start.month / 12),
                month_cos=np.cos(2 * np.pi * start.month / 12),
                **ss.loc[start].to_dict(),
                prev_log_gap=np.log((start - dates[i - 1]).days) if i > 0 else np.nan,
                log_n_prev=np.log1p(i),
                log_alt=s.log_alt,
                **{c: s[c] for c in s.index if c.startswith("orbit_")},
                age_years=(start - s.launch_date).days / 365.25 if pd.notna(s.launch_date) else np.nan,
                inclination=s.inclination, perigee=s.perigee, apogee=s.apogee,
            ))
    return pd.DataFrame(rows)


def add_label(tte: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Binary target: 1 if the next anomaly occurs within `horizon` days, else 0.
    Censored gaps shorter than the horizon are dropped (we cannot know the outcome)."""
    keep = (tte.event == 1) | (tte.duration >= horizon)
    out = tte[keep].copy()
    out["label"] = ((out.event == 1) & (out.duration <= horizon)).astype(int)
    return out.reset_index(drop=True)
