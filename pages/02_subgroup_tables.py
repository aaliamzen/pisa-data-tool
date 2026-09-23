import streamlit as st
import pandas as pd
import numpy as np
import re
from scipy.stats import t

try:
    st.logo("assets/logo.png")
except Exception as e:
    st.error(f"Failed to load logo: {e}")

st.set_page_config(page_title="Subgroup Tables - PISA Data Exploration Tool", layout="wide")

if "df" not in st.session_state or st.session_state.df is None:
    st.error("Please upload your data first in the Data Upload page!")
    st.stop()

df = st.session_state.df
variable_labels = st.session_state.get("variable_labels", {})
value_labels = st.session_state.get("value_labels", {})
visible_columns = st.session_state.get("visible_columns", list(df.columns))

if "W_FSTUWT" not in df.columns:
    st.error("The final student weight (W_FSTUWT) is not present in your data.")
    st.stop()

replicate_weight_cols = [f"W_FSTURWT{i}" for i in range(1, 81)]
missing_weights = [c for c in replicate_weight_cols if c not in df.columns]
if missing_weights:
    st.error(
        "Replicate weights W_FSTURWT1-W_FSTURWT80 are required. "
        "Missing {0} column(s), e.g. {1}.".format(len(missing_weights), missing_weights[:5])
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
domain_options = [domain_to_label[d] for d in pv_domains if d in domain_to_label]
label_to_domain = {domain_to_label[d]: d for d in pv_domains if d in domain_to_label}

st.title("Subgroup tables")
st.markdown(
    """
**How to use this page**
- **Mean** of a score or scale, or **percentage** (a categorical item, or % below Level 2 / at Levels 5–6 for a score domain).
- **Group by** is always the categorical split (gender, ESCS tertiles, school type, …).
- Group by gender, school type if present, another short categorical, or **ESCS tertiles** built on this country file.
- Each row is the group estimate and Fay-BRR SE. The gap versus the reference group also has a design-based SE.
- Score domains use 10 plausible values and Rubin's rules. ESCS tertile cuts are weighted and then held fixed across replicates.
- This is for exploration. Official OECD “by gender” tables can still differ slightly because of recodes and rounding.
"""
)
st.caption("Using 80 BRR replicate weights (Fay k = 0.5).")


def weighted_mean(x, w):
    s = np.sum(w)
    if s == 0:
        return np.nan
    return float(np.sum(x * w) / s)


def weighted_percent(ind, w):
    s = np.sum(w)
    if s == 0:
        return np.nan
    return float(100.0 * np.sum(ind * w) / s)


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
    m = np.nanmean(vals)
    if k == 1:
        return m, np.nanmean(ses)
    within = np.nanmean(ses ** 2)
    between = np.nanvar(vals, ddof=1)
    return m, float(np.sqrt(within + (1.0 + 1.0 / k) * between))


def pretty_cat(col, val):
    if col in value_labels and val in value_labels[col]:
        return str(value_labels[col][val])
    if col in value_labels:
        try:
            return str(value_labels[col].get(int(val), val))
        except Exception:
            pass
    return str(val)


def weighted_tertiles(series, weights):
    mask = series.notna() & weights.notna()
    x = series.loc[mask].astype(float).values
    w = weights.loc[mask].astype(float).values
    order = np.argsort(x)
    x = x[order]
    w = w[order]
    cw = np.cumsum(w)
    tot = cw[-1]
    if tot <= 0:
        return None, None
    q1 = x[np.searchsorted(cw, tot / 3.0, side="left")]
    q2 = x[np.searchsorted(cw, 2.0 * tot / 3.0, side="left")]
    cuts = (float(q1), float(q2))

    def assign(s):
        out = pd.Series(np.nan, index=s.index, dtype=object)
        ok = s.notna()
        v = s.loc[ok].astype(float)
        labs = np.where(v <= cuts[0], "Low ESCS (bottom third)",
                        np.where(v <= cuts[1], "Middle ESCS", "High ESCS (top third)"))
        out.loc[ok] = labs
        return out

    return assign, cuts


def is_cat_series(s, max_unique=12):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s):
        return True
    return s.nunique(dropna=True) <= max_unique


# Grouping options
group_built_in = []
if "ESCS" in df.columns:
    group_built_in.append("ESCS tertiles (this country, weighted)")

preferred_groups = []
for code in ["ST004D01T", "MALE", "SCHLTYPE", "SC013Q01TA", "PRIVATESCH", "PROGN"]:
    if code in df.columns:
        lab = variable_labels.get(code, code)
        preferred_groups.append((lab if lab != code else code, code))

cat_visible = []
for code in visible_columns:
    if code not in df.columns or code in ("W_FSTUWT",) or code.startswith("W_FSTURWT"):
        continue
    if pv_pattern.match(str(code)):
        continue
    if is_cat_series(df[code]):
        lab = variable_labels.get(code, code)
        name = lab if lab not in [p[0] for p in preferred_groups] else f"{lab} ({code})"
        cat_visible.append((name, code))

group_choices = group_built_in + [p[0] for p in preferred_groups]
seen = set(group_choices)
for name, code in cat_visible:
    if name not in seen:
        group_choices.append(name)
        seen.add(name)
name_to_code = {p[0]: p[1] for p in preferred_groups + cat_visible}

# Outcome options
num_labels = []
num_map = {}
for code in visible_columns:
    if code not in df.columns or pv_pattern.match(str(code)):
        continue
    if df[code].dtype not in ["float64", "int64", "float32", "int32"]:
        continue
    if is_cat_series(df[code], max_unique=10):
        continue
    lab = variable_labels.get(code, code)
    name = lab if lab not in num_map else f"{lab} ({code})"
    num_labels.append(name)
    num_map[name] = code

cat_out_labels = []
cat_out_map = {}
for code in visible_columns:
    if code not in df.columns or pv_pattern.match(str(code)):
        continue
    if is_cat_series(df[code], max_unique=20):
        lab = variable_labels.get(code, code)
        name = lab if lab not in cat_out_map else f"{lab} ({code})"
        cat_out_labels.append(name)
        cat_out_map[name] = code

outcome_kind = st.radio(
    "What to tabulate",
    ["Mean of a score or scale", "Percentage (category or proficiency)"],
    horizontal=True,
    help="Group by is always categorical. Percentage outcomes are questionnaire categories, or official PISA proficiency shares if you pick a score domain.",
)

if outcome_kind.startswith("Mean"):
    outcome_opts = domain_options + num_labels
else:
    outcome_opts = domain_options + cat_out_labels

c1, c2 = st.columns(2)
with c1:
    outcome_lab = st.selectbox("Outcome", [""] + outcome_opts)
with c2:
    group_lab = st.selectbox("Group by", [""] + group_choices)

run = st.button("Run subgroup table")

if run and outcome_lab and group_lab:
    work = df.copy()
    # Group assignment
    if group_lab.startswith("ESCS tertiles"):
        assign, cuts = weighted_tertiles(work["ESCS"], work["W_FSTUWT"])
        if assign is None:
            st.error("Could not form ESCS tertiles.")
            st.stop()
        work["_grp"] = assign(work["ESCS"])
        group_title = "ESCS tertiles (weighted cuts {0:.2f}, {1:.2f})".format(cuts[0], cuts[1])
        ref_default = "Low ESCS (bottom third)"
    else:
        gcode = name_to_code[group_lab]
        work["_grp"] = work[gcode].map(lambda v: pretty_cat(gcode, v) if pd.notna(v) else np.nan)
        group_title = group_lab
        ref_default = None

    groups = [g for g in work["_grp"].dropna().unique().tolist()]
    # Stable order: Low/Middle/High, Female/Male, else sorted
    def sort_key(g):
        gl = str(g).lower()
        if "low" in gl:
            return (0, gl)
        if "middle" in gl:
            return (1, gl)
        if "high" in gl:
            return (2, gl)
        if "female" in gl:
            return (0, gl)
        if "male" in gl:
            return (1, gl)
        return (5, gl)

    groups = sorted(groups, key=sort_key)
    if not groups:
        st.error("No group values after missing data.")
        st.stop()

    ref = st.selectbox("Reference group for gaps", groups, index=groups.index(ref_default) if ref_default in groups else 0)

    if outcome_kind.startswith("Mean"):
        if outcome_lab in label_to_domain:
            ycols = pv_domains[label_to_domain[outcome_lab]]
            st.caption("Plausible values: {0} ({1} PVs) + Rubin.".format(outcome_lab, len(ycols)))
        else:
            ycols = [num_map[outcome_lab]]
            st.caption("No score domain. Single observed variable.")

        rows = []
        means_main = {}
        ses_main = {}
        ns = {}
        for g in groups:
            pv_m, pv_se, pv_n = [], [], []
            for ycol in ycols:
                sub = work.loc[work["_grp"] == g, [ycol, "W_FSTUWT"] + replicate_weight_cols].dropna()
                if len(sub) < 2:
                    pv_m.append(np.nan)
                    pv_se.append(np.nan)
                    pv_n.append(len(sub))
                    continue
                main = weighted_mean(sub[ycol].values, sub["W_FSTUWT"].values)
                reps = [weighted_mean(sub[ycol].values, sub[rw].values) for rw in replicate_weight_cols]
                pv_m.append(main)
                pv_se.append(fay_se(main, reps))
                pv_n.append(len(sub))
            m, se = rubin(pv_m, pv_se)
            means_main[g] = m
            ses_main[g] = se
            ns[g] = int(np.nanmean(pv_n)) if pv_n else 0

        # Gaps vs ref: difference on each PV / replicate then Rubin
        gap_m, gap_se = {}, {}
        for g in groups:
            if g == ref:
                gap_m[g], gap_se[g] = 0.0, np.nan
                continue
            pv_d, pv_dse = [], []
            for ycol in ycols:
                diffs = []
                main_diff = np.nan
                for wi, wcol in enumerate(["W_FSTUWT"] + replicate_weight_cols):
                    a = work.loc[work["_grp"] == g, [ycol, wcol]].dropna()
                    b = work.loc[work["_grp"] == ref, [ycol, wcol]].dropna()
                    if len(a) < 2 or len(b) < 2:
                        d = np.nan
                    else:
                        d = weighted_mean(a[ycol].values, a[wcol].values) - weighted_mean(b[ycol].values, b[wcol].values)
                    if wi == 0:
                        main_diff = d
                    else:
                        diffs.append(d)
                pv_d.append(main_diff)
                pv_dse.append(fay_se(main_diff, diffs))
            gap_m[g], gap_se[g] = rubin(pv_d, pv_dse)

        table_rows = []
        for g in groups:
            se = ses_main[g]
            gm = means_main[g]
            gd, gse = gap_m[g], gap_se[g]
            if g == ref:
                p = np.nan
            elif gse and not np.isnan(gse) and gse > 0 and not np.isnan(gd):
                p = 2 * (1 - t.cdf(abs(gd / gse), df=max(ns[g] + ns[ref] - 2, 1)))
            else:
                p = np.nan
            star = "**" if (not np.isnan(p) and p < 0.001) else "*" if (not np.isnan(p) and p < 0.01) else ""
            table_rows.append({
                "Group": g,
                "N": ns[g],
                "Mean": "-" if np.isnan(gm) else f"{gm:.2f}",
                "SE": "-" if np.isnan(se) else f"{se:.2f}",
                "Gap vs {0}".format(ref): "0.00 (ref)" if g == ref else ("-" if np.isnan(gd) else f"{gd:.2f}{star}"),
                "SE(gap)": "-" if (g == ref or np.isnan(gse)) else f"{gse:.2f}",
                "p (gap)": "-" if np.isnan(p) else ("< .001" if p < 0.001 else f"{p:.3f}"),
            })
        out = pd.DataFrame(table_rows)
        st.markdown("**{0}** by **{1}**".format(outcome_lab, group_title))
        st.dataframe(out, hide_index=True, use_container_width=True)
        st.caption(
            "Note. Weighted with W_FSTUWT. SE and SE(gap) are Fay BRR (k = 0.5). "
            "Gaps are this group minus the reference. *p < .01. **p < .001. "
            "N is unweighted cases in the group with non-missing outcome."
        )
    elif outcome_lab in label_to_domain:
        BASELINE = {"SCIE": 409.54, "READ": 407.47, "MATH": 420.07}
        TOP5 = {"SCIE": 633.33, "READ": 625.61, "MATH": 606.99}
        dcode = label_to_domain[outcome_lab]
        ycols = pv_domains[dcode]
        shares = [
            ("Below Level 2", None, BASELINE[dcode]),
            ("Level 5 or 6", TOP5[dcode], None),
        ]
        st.caption("Proficiency shares from official PISA 2025 cuts. 10 PVs + Rubin.")

        def share_fn(x, w, low, high):
            x = np.asarray(x, dtype=float)
            w = np.asarray(w, dtype=float)
            ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
            x, w = x[ok], w[ok]
            if len(x) == 0 or w.sum() == 0:
                return np.nan
            if low is None:
                hit = x < high
            elif high is None:
                hit = x >= low
            else:
                hit = (x >= low) & (x < high)
            return float(100.0 * np.sum(w[hit]) / np.sum(w))

        blocks = []
        for slabel, low, high in shares:
            rows = []
            for g in groups:
                pv_m, pv_se, pv_n = [], [], []
                for ycol in ycols:
                    sub = work.loc[work["_grp"] == g, [ycol, "W_FSTUWT"] + replicate_weight_cols].dropna()
                    if len(sub) < 2:
                        pv_m.append(np.nan)
                        pv_se.append(np.nan)
                        pv_n.append(len(sub))
                        continue
                    main = share_fn(sub[ycol].values, sub["W_FSTUWT"].values, low, high)
                    reps = [share_fn(sub[ycol].values, sub[rw].values, low, high) for rw in replicate_weight_cols]
                    pv_m.append(main)
                    pv_se.append(fay_se(main, reps))
                    pv_n.append(len(sub))
                m, se = rubin(pv_m, pv_se)
                if g == ref:
                    gd, gse, p = 0.0, np.nan, np.nan
                else:
                    pv_d, pv_dse = [], []
                    for ycol in ycols:
                        diffs = []
                        main_d = np.nan
                        for wi, wcol in enumerate(["W_FSTUWT"] + replicate_weight_cols):
                            a = work.loc[work["_grp"] == g, [ycol, wcol]].dropna()
                            b = work.loc[work["_grp"] == ref, [ycol, wcol]].dropna()
                            if len(a) < 2 or len(b) < 2:
                                d = np.nan
                            else:
                                d = share_fn(a[ycol].values, a[wcol].values, low, high) - share_fn(
                                    b[ycol].values, b[wcol].values, low, high
                                )
                            if wi == 0:
                                main_d = d
                            else:
                                diffs.append(d)
                        pv_d.append(main_d)
                        pv_dse.append(fay_se(main_d, diffs))
                    gd, gse = rubin(pv_d, pv_dse)
                    p = (
                        2 * (1 - t.cdf(abs(gd / gse), df=max(int(np.nanmean(pv_n)) + 2, 1)))
                        if gse and gse > 0 and not np.isnan(gd)
                        else np.nan
                    )
                star = "**" if (not np.isnan(p) and p < 0.001) else "*" if (not np.isnan(p) and p < 0.01) else ""
                rows.append({
                    "Share": slabel,
                    "Group": g,
                    "N": int(np.nanmean(pv_n)) if pv_n else 0,
                    "%": "-" if np.isnan(m) else f"{m:.1f}",
                    "SE": "-" if np.isnan(se) else f"{se:.2f}",
                    "Gap vs {0} (pp)".format(ref): "0.0 (ref)" if g == ref else ("-" if np.isnan(gd) else f"{gd:.1f}{star}"),
                    "SE(gap)": "-" if (g == ref or np.isnan(gse)) else f"{gse:.2f}",
                    "p (gap)": "-" if np.isnan(p) else ("< .001" if p < 0.001 else f"{p:.3f}"),
                })
            blocks.append(pd.DataFrame(rows))
        out = pd.concat(blocks, ignore_index=True)
        st.markdown("**{0} proficiency shares** by **{1}**".format(outcome_lab, group_title))
        st.dataframe(out, hide_index=True, use_container_width=True)
        st.caption(
            "Note. Weighted proficiency shares. SE and SE(gap) are Fay BRR (k = 0.5) with Rubin's rules. "
            "*p < .01. **p < .001."
        )
    else:
        ocode = cat_out_map[outcome_lab]
        cats = [pretty_cat(ocode, v) for v in work[ocode].dropna().unique()]
        work["_out"] = work[ocode].map(lambda v: pretty_cat(ocode, v) if pd.notna(v) else np.nan)
        focus = st.multiselect("Show percentage in these outcome categories", cats, default=cats[: min(4, len(cats))])
        if not focus:
            st.stop()
        blocks = []
        for cat in focus:
            rows = []
            pct_main, se_main, nn = {}, {}, {}
            for g in groups:
                sub = work.loc[work["_grp"] == g, ["_out", "W_FSTUWT"] + replicate_weight_cols].dropna(subset=["_out", "W_FSTUWT"])
                if len(sub) < 2:
                    pct_main[g] = np.nan
                    se_main[g] = np.nan
                    nn[g] = len(sub)
                    continue
                ind = (sub["_out"] == cat).astype(float).values
                main = weighted_percent(ind, sub["W_FSTUWT"].values)
                reps = []
                for rw in replicate_weight_cols:
                    sw = sub[[ "_out", rw]].dropna()
                    reps.append(weighted_percent((sw["_out"] == cat).astype(float).values, sw[rw].values))
                pct_main[g] = main
                se_main[g] = fay_se(main, reps)
                nn[g] = int((work["_grp"] == g).sum())
            for g in groups:
                # gap
                if g == ref:
                    gd, gse, p = 0.0, np.nan, np.nan
                else:
                    diffs = []
                    main_d = np.nan
                    for wi, wcol in enumerate(["W_FSTUWT"] + replicate_weight_cols):
                        a = work.loc[work["_grp"] == g, ["_out", wcol]].dropna()
                        b = work.loc[work["_grp"] == ref, ["_out", wcol]].dropna()
                        if len(a) < 2 or len(b) < 2:
                            d = np.nan
                        else:
                            d = weighted_percent((a["_out"] == cat).astype(float).values, a[wcol].values) - weighted_percent(
                                (b["_out"] == cat).astype(float).values, b[wcol].values
                            )
                        if wi == 0:
                            main_d = d
                        else:
                            diffs.append(d)
                    gd = main_d
                    gse = fay_se(main_d, diffs)
                    p = 2 * (1 - t.cdf(abs(gd / gse), df=max(nn.get(g, 2) + nn.get(ref, 2) - 2, 1))) if gse and gse > 0 and not np.isnan(gd) else np.nan
                star = "**" if (not np.isnan(p) and p < 0.001) else "*" if (not np.isnan(p) and p < 0.01) else ""
                rows.append({
                    "Outcome category": cat,
                    "Group": g,
                    "N": nn.get(g, 0),
                    "%": "-" if np.isnan(pct_main.get(g, np.nan)) else f"{pct_main[g]:.1f}",
                    "SE": "-" if np.isnan(se_main.get(g, np.nan)) else f"{se_main[g]:.1f}",
                    "Gap vs {0} (pp)".format(ref): "0.0 (ref)" if g == ref else ("-" if np.isnan(gd) else f"{gd:.1f}{star}"),
                    "SE(gap)": "-" if (g == ref or np.isnan(gse)) else f"{gse:.1f}",
                    "p (gap)": "-" if np.isnan(p) else ("< .001" if p < 0.001 else f"{p:.3f}"),
                })
            blocks.append(pd.DataFrame(rows))
        out = pd.concat(blocks, ignore_index=True)
        st.markdown("**{0}** by **{1}**".format(outcome_lab, group_title))
        st.dataframe(out, hide_index=True, use_container_width=True)
        st.caption(
            "Note. Weighted percentages. SE and SE(gap) are Fay BRR (k = 0.5). "
            "Gap is percentage points versus the reference group. *p < .01. **p < .001."
        )

# Instructions are shown under the page title.
