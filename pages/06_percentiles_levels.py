import streamlit as st
import pandas as pd
import numpy as np
import re

try:
    st.logo("assets/logo.png")
except Exception as e:
    st.error(f"Failed to load logo: {e}")

st.set_page_config(page_title="Percentiles and proficiency levels - PISA", layout="wide")

if "df" not in st.session_state or st.session_state.df is None:
    st.error("Please upload your data first in the Data Upload page!")
    st.stop()

df = st.session_state.df

if "W_FSTUWT" not in df.columns:
    st.error("The final student weight (W_FSTUWT) is not present in your data.")
    st.stop()

replicate_weight_cols = [f"W_FSTURWT{i}" for i in range(1, 81)]
missing_weights = [c for c in replicate_weight_cols if c not in df.columns]
if missing_weights:
    st.error(
        "Replicate weights W_FSTURWT1-W_FSTURWT80 are required. Missing {0} column(s).".format(
            len(missing_weights)
        )
    )
    st.stop()

label = st.session_state.get("dataset_label")
if label:
    st.info(f"Dataset: {label}")

pv_pattern = re.compile(r"^PV([1-9]|10)(MATH|READ|SCIE)(\d*)$", re.IGNORECASE)
domain_to_label = {"MATH": "Mathematics score", "READ": "Reading score", "SCIE": "Science score"}
pv_domains = {}
for col in df.columns:
    m = pv_pattern.match(str(col))
    if m:
        pv_domains.setdefault(m.group(2).upper(), []).append(col)
for d in pv_domains:
    pv_domains[d].sort(key=lambda x: int(re.match(r"PV(\d+)", x, re.IGNORECASE).group(1)))

# Official PISA 2025 lower bounds (NCES / OECD technical notes)
# Each tuple: (label, lower inclusive)
LEVEL_CUTS = {
    "SCIE": [
        ("Below Level 1b", None),
        ("Level 1b", 260.54),
        ("Level 1a", 334.94),
        ("Level 2", 409.54),
        ("Level 3", 484.14),
        ("Level 4", 558.73),
        ("Level 5", 633.33),
        ("Level 6", 707.93),
    ],
    "READ": [
        ("Below Level 1c", None),
        ("Level 1c", 189.33),
        ("Level 1b", 262.04),
        ("Level 1a", 334.75),
        ("Level 2", 407.47),
        ("Level 3", 480.18),
        ("Level 4", 552.89),
        ("Level 5", 625.61),
        ("Level 6", 698.32),
    ],
    "MATH": [
        ("Below Level 1c", None),
        ("Level 1c", 233.17),
        ("Level 1b", 295.47),
        ("Level 1a", 357.77),
        ("Level 2", 420.07),
        ("Level 3", 482.38),
        ("Level 4", 544.68),
        ("Level 5", 606.99),
        ("Level 6", 669.30),
    ],
}

BASELINE = {"SCIE": 409.54, "READ": 407.47, "MATH": 420.07}
TOP5 = {"SCIE": 633.33, "READ": 625.61, "MATH": 606.99}

st.title("Percentiles and proficiency levels")
st.markdown(
    """
**How to use this page**
- Choose Mathematics, Reading or Science score, then **Run analysis**.
- Percentiles (P5 to P95 and P90−P10) describe the score distribution.
- Proficiency levels use official PISA 2025 cut-scores. Level 2 is the OECD baseline; Levels 5–6 are top performers.
- Estimates use `W_FSTUWT`. SEs use Fay BRR (*k* = 0.5) and Rubin's rules across 10 plausible values.
"""
)
st.caption("Using 80 BRR replicate weights (Fay k = 0.5). Cut-scores from the PISA 2025 technical notes.")

available = [d for d in ("SCIE", "READ", "MATH") if d in pv_domains and len(pv_domains[d]) >= 1]
if not available:
    st.error("No mathematics / reading / science plausible values found in this file.")
    st.stop()

domain_lab = st.selectbox(
    "Score domain",
    [domain_to_label[d] for d in available],
)
domain = {v: k for k, v in domain_to_label.items()}[domain_lab]
pv_cols = pv_domains[domain]
st.info("{0}: {1} plausible values.".format(domain_lab, len(pv_cols)))

PCTS = [5, 10, 25, 50, 75, 90, 95]


def fay_se(main, reps):
    reps = np.array([r for r in reps if not np.isnan(r)], dtype=float)
    if len(reps) == 0 or np.isnan(main):
        return np.nan
    return float(np.sqrt((1.0 / 20.0) * np.sum((reps - main) ** 2)))


def rubin(vals, ses):
    vals = np.array(vals, dtype=float)
    ses = np.array(ses, dtype=float)
    k = int(np.sum(~np.isnan(vals)))
    if k == 0:
        return np.nan, np.nan
    m = float(np.nanmean(vals))
    if k == 1:
        return m, float(np.nanmean(ses))
    within = float(np.nanmean(ses ** 2))
    between = float(np.nanvar(vals, ddof=1))
    return m, float(np.sqrt(within + (1.0 + 1.0 / k) * between))


def weighted_percentile(x, w, p):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x = x[mask]
    w = w[mask]
    if len(x) == 0:
        return np.nan
    order = np.argsort(x)
    x = x[order]
    w = w[order]
    cw = np.cumsum(w)
    target = (p / 100.0) * cw[-1]
    idx = int(np.searchsorted(cw, target, side="left"))
    idx = min(idx, len(x) - 1)
    return float(x[idx])


def weighted_share(x, w, low, high):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x = x[mask]
    w = w[mask]
    if len(x) == 0 or w.sum() == 0:
        return np.nan
    if low is None and high is None:
        hit = np.ones(len(x), dtype=bool)
    elif low is None:
        hit = x < high
    elif high is None:
        hit = x >= low
    else:
        hit = (x >= low) & (x < high)
    return float(100.0 * np.sum(w[hit]) / np.sum(w))


def brr_stat(work, ycol, fn):
    cols = [ycol, "W_FSTUWT"] + replicate_weight_cols
    data = work[cols].dropna()
    if len(data) < 2:
        return np.nan, np.nan, len(data)
    main = fn(data[ycol].values, data["W_FSTUWT"].values)
    reps = [fn(data[ycol].values, data[rw].values) for rw in replicate_weight_cols]
    return main, fay_se(main, reps), len(data)


run = st.button("Run analysis")

if run:
    work = df
    # Percentiles
    pct_rows = []
    for p in PCTS:
        pv_v, pv_se = [], []
        n_used = []
        for ycol in pv_cols:
            main, se, n = brr_stat(work, ycol, lambda x, w, pp=p: weighted_percentile(x, w, pp))
            pv_v.append(main)
            pv_se.append(se)
            n_used.append(n)
        m, se = rubin(pv_v, pv_se)
        pct_rows.append({
            "Percentile": "P{0}".format(p),
            "Score": "-" if np.isnan(m) else f"{m:.1f}",
            "SE": "-" if np.isnan(se) else f"{se:.2f}",
        })
    # P90-P10
    def p90m10(x, w):
        return weighted_percentile(x, w, 90) - weighted_percentile(x, w, 10)

    pv_v, pv_se = [], []
    for ycol in pv_cols:
        main, se, n = brr_stat(work, ycol, p90m10)
        pv_v.append(main)
        pv_se.append(se)
    spread, spread_se = rubin(pv_v, pv_se)
    pct_rows.append({
        "Percentile": "P90 − P10",
        "Score": "-" if np.isnan(spread) else f"{spread:.1f}",
        "SE": "-" if np.isnan(spread_se) else f"{spread_se:.2f}",
    })
    st.subheader("Score percentiles")
    st.dataframe(pd.DataFrame(pct_rows), hide_index=True, use_container_width=True)
    st.caption("N (unweighted, first PV non-missing) = {0:,}.".format(int(np.nanmean(n_used)) if n_used else 0))

    # Proficiency levels
    cuts = LEVEL_CUTS[domain]
    level_rows = []
    for i, (lab, low) in enumerate(cuts):
        high = cuts[i + 1][1] if i + 1 < len(cuts) else None
        pv_v, pv_se = [], []
        for ycol in pv_cols:
            main, se, n = brr_stat(
                work, ycol, lambda x, w, lo=low, hi=high: weighted_share(x, w, lo, hi)
            )
            pv_v.append(main)
            pv_se.append(se)
        m, se = rubin(pv_v, pv_se)
        level_rows.append({
            "Proficiency level": lab,
            "Lower cut": "—" if low is None else f"{low:.2f}",
            "%": "-" if np.isnan(m) else f"{m:.1f}",
            "SE": "-" if np.isnan(se) else f"{se:.2f}",
        })

    # Below Level 2 and Level 5+
    summaries = [
        ("Below Level 2 (low performers)", None, BASELINE[domain]),
        ("Level 2 or above (baseline+)", BASELINE[domain], None),
        ("Level 5 or 6 (top performers)", TOP5[domain], None),
    ]
    sum_rows = []
    for lab, low, high in summaries:
        pv_v, pv_se = [], []
        for ycol in pv_cols:
            main, se, n = brr_stat(
                work, ycol, lambda x, w, lo=low, hi=high: weighted_share(x, w, lo, hi)
            )
            pv_v.append(main)
            pv_se.append(se)
        m, se = rubin(pv_v, pv_se)
        sum_rows.append({
            "Summary": lab,
            "%": "-" if np.isnan(m) else f"{m:.1f}",
            "SE": "-" if np.isnan(se) else f"{se:.2f}",
        })

    st.subheader("Proficiency levels")
    st.dataframe(pd.DataFrame(level_rows), hide_index=True, use_container_width=True)
    st.subheader("Headline shares")
    st.dataframe(pd.DataFrame(sum_rows), hide_index=True, use_container_width=True)
    st.caption(
        "Note. Weighted with W_FSTUWT. SE uses Fay BRR (k = 0.5) and Rubin's rules across 10 PVs. "
        "Cut-scores are the official PISA 2025 lower bounds (OECD / NCES technical notes). "
        "Level 2 is the OECD baseline. Shares may not sum to 100.0 because of rounding."
    )

# Instructions are shown under the page title.
