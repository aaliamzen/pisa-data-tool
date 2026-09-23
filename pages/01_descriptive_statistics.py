import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

try:
    st.logo("assets/logo.png")
except Exception as e:
    st.error(f"Failed to load logo: {e}")

st.set_page_config(page_title="PISA Data Exploration Tool - Descriptive Statistics", layout="wide")

if "df" not in st.session_state or st.session_state.df is None:
    st.error("Please upload your data first in the Data Upload page!")
    st.stop()

df = st.session_state.df
variable_labels = st.session_state.variable_labels
value_labels = st.session_state.value_labels
visible_columns = st.session_state.visible_columns

if "W_FSTUWT" not in df.columns:
    st.error("The final student weight (W_FSTUWT) is not present in your data. Weighted statistics cannot be computed.")
    st.stop()

replicate_weight_cols = [f"W_FSTURWT{i}" for i in range(1, 81)]
missing_weights = [col for col in replicate_weight_cols if col not in df.columns]
if missing_weights:
    st.error(
        "Replicate weights W_FSTURWT1-W_FSTURWT80 are required for standard errors. "
        "Missing {0} column(s), e.g. {1}. Re-export the file with student replicate weights.".format(
            len(missing_weights), missing_weights[:5]
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
        domain = m.group(2).upper()
        pv_domains.setdefault(domain, []).append(col)
for domain in pv_domains:
    pv_domains[domain].sort(key=lambda x: int(re.match(r"PV(\d+)", x, re.IGNORECASE).group(1)))

domain_options = [domain_to_label[d] for d in pv_domains if d in domain_to_label]
label_to_domain = {domain_to_label[d]: d for d in pv_domains if d in domain_to_label}

var_code_to_label = {code: variable_labels.get(code, code) for code in visible_columns if code in df.columns}
# Avoid listing individual PVs separately when the domain is offered
var_code_to_label = {
    code: lab
    for code, lab in var_code_to_label.items()
    if not pv_pattern.match(str(code))
}
var_label_to_code = {}
unique_labels = []
seen = {}
for code, lab in var_code_to_label.items():
    name = lab if lab not in seen else f"{lab} ({code})"
    seen[lab] = True
    unique_labels.append(name)
    var_label_to_code[name] = code

st.title("Descriptive Statistics")
st.markdown(
    """
**How to use this page**
- Select scales and/or score domains (Mathematics, Reading, Science score).
- Continuous results: weighted mean, Fay-BRR standard error, SD, skewness and kurtosis.
- Categorical results: weighted % in each category with a Fay-BRR SE.
- Score domains use all 10 plausible values and Rubin's rules for the mean and its SE.
- Means and percentages use `W_FSTUWT`. SEs use Fay BRR (*k* = 0.5) on 80 replicate weights.
"""
)
st.caption("Using 80 BRR replicate weights (Fay k = 0.5).")

all_options = domain_options + unique_labels
selected_labels = st.multiselect(
    "Select Variables for Analysis",
    options=all_options,
    default=[],
    help="Choose scales or score domains (Mathematics / Reading / Science score).",
)

selected_domains = [label_to_domain[lab] for lab in selected_labels if lab in label_to_domain]
selected_codes = [var_label_to_code[lab] for lab in selected_labels if lab in var_label_to_code]
if selected_domains:
    st.info(
        "Score domains will use all plausible values and Rubin's rules: "
        + "; ".join(
            "{0} ({1} PVs)".format(domain_to_label[d], len(pv_domains.get(d, [])))
            for d in selected_domains
        )
    )


def weighted_mean(x, w):
    s = np.sum(w)
    if s == 0:
        return np.nan
    return np.sum(x * w) / s


def weighted_std(x, w):
    m = weighted_mean(x, w)
    s = np.sum(w)
    if s == 0 or np.isnan(m):
        return np.nan
    return np.sqrt(np.sum(w * (x - m) ** 2) / s)


def weighted_skew(x, w):
    m = weighted_mean(x, w)
    sd = weighted_std(x, w)
    s = np.sum(w)
    if s == 0 or sd == 0 or np.isnan(sd):
        return np.nan
    return np.sum(w * ((x - m) / sd) ** 3) / s


def weighted_kurtosis(x, w):
    m = weighted_mean(x, w)
    sd = weighted_std(x, w)
    s = np.sum(w)
    if s == 0 or sd == 0 or np.isnan(sd):
        return np.nan
    return np.sum(w * ((x - m) / sd) ** 4) / s


def is_categorical(series):
    if pd.api.types.is_categorical_dtype(series) or pd.api.types.is_object_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return series.nunique(dropna=True) <= 10
    return False


def fay_brr_se(main, replicates):
    reps = np.array([r for r in replicates if not np.isnan(r)], dtype=float)
    if len(reps) == 0 or np.isnan(main):
        return np.nan
    return np.sqrt((1.0 / 20.0) * np.sum((reps - main) ** 2))


def brr_se_mean(work, col):
    cols = [col, "W_FSTUWT"] + replicate_weight_cols
    data = work[cols].dropna()
    if len(data) < 2:
        return np.nan, len(data)
    main = weighted_mean(data[col].values, data["W_FSTUWT"].values)
    reps = []
    for rw in replicate_weight_cols:
        reps.append(weighted_mean(data[col].values, data[rw].values))
    return fay_brr_se(main, reps), len(data)


def brr_se_percent(work, col, category):
    cols = [col, "W_FSTUWT"] + replicate_weight_cols
    data = work[cols].dropna()
    if len(data) < 2:
        return np.nan
    w = data["W_FSTUWT"].values
    ind = (data[col].values == category).astype(float)
    tw = np.sum(w)
    if tw == 0:
        return np.nan
    main = 100.0 * np.sum(w * ind) / tw
    reps = []
    for rw in replicate_weight_cols:
        wr = data[rw].values
        s = np.sum(wr)
        if s == 0:
            continue
        reps.append(100.0 * np.sum(wr * ind) / s)
    return fay_brr_se(main, reps)


def rubin_mean(means, ses):
    means = np.array(means, dtype=float)
    ses = np.array(ses, dtype=float)
    m = np.nanmean(means)
    valid = ~np.isnan(means) & ~np.isnan(ses)
    k = int(np.sum(valid))
    if k == 0:
        return np.nan, np.nan
    within = np.nanmean(ses[valid] ** 2)
    between = np.nanvar(means[valid], ddof=1) if k > 1 else 0.0
    total = within + (1.0 + 1.0 / k) * between
    return m, np.sqrt(total)


def create_apa_table_html(frame, title, note):
    css = """
    <style>
        .apa-table { font-family: "Times New Roman", Times, serif; font-size: 12pt; border-collapse: collapse; width: 100%; }
        .apa-table caption { font-style: italic; text-align: left; margin-bottom: 10px; }
        .apa-table th { border-top: 1px solid black; border-bottom: 1px solid black; font-weight: normal; padding: 8px; text-align: center; }
        .apa-table td { padding: 8px; }
        .apa-table td.left { text-align: left; }
        .apa-table td.numeric { text-align: center; }
        .apa-table tr:last-child td { border-bottom: 1px solid black; }
        .table-number { font-family: "Times New Roman", Times, serif; font-weight: bold; text-align: left; padding-bottom: 8px; }
        .table-note { font-family: "Times New Roman", Times, serif; font-size: 12pt; text-align: left; margin-top: 10px; }
    </style>
    """
    html = css
    html += '<div class="table-number">Table 1</div>'
    html += '<table class="apa-table">'
    html += f"<caption>{title}</caption><thead><tr>"
    for col in frame.columns:
        if col in ["N", "M", "SD", "Mean", "SE"]:
            html += f"<th><em>{col}</em></th>"
        else:
            html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>"
    for _, row in frame.iterrows():
        html += "<tr>"
        for idx, col in enumerate(frame.columns):
            val = row[col]
            if pd.isna(val) or val == "":
                cell = "-"
            else:
                cell = str(val)
            cls = "left" if idx == 0 else "numeric"
            html += f"<td class='{cls}'>{cell}</td>"
        html += "</tr>"
    html += f'</tbody></table><div class="table-note"><em>Note.</em> {note}</div>'
    return html


if selected_labels:
    numeric_rows = []
    cat_frames = []
    plot_items = []  # (label, series or None for domain first pv, is_cat)

    for lab in selected_labels:
        if lab in label_to_domain:
            domain = label_to_domain[lab]
            pv_list = pv_domains.get(domain, [])
            means, ses, ns = [], [], []
            sds, skews, kurts = [], [], []
            for pv in pv_list:
                mask = df[pv].notna() & df["W_FSTUWT"].notna()
                x = df.loc[mask, pv].astype(float).values
                w = df.loc[mask, "W_FSTUWT"].values
                if len(x) == 0:
                    continue
                means.append(weighted_mean(x, w))
                se, n = brr_se_mean(df, pv)
                ses.append(se)
                ns.append(n)
                sds.append(weighted_std(x, w))
                skews.append(weighted_skew(x, w))
                kurts.append(weighted_kurtosis(x, w))
            comb_m, comb_se = rubin_mean(means, ses)
            numeric_rows.append({
                "Variable": lab,
                "N": int(np.nanmean(ns)) if ns else 0,
                "Mean": "-" if np.isnan(comb_m) else f"{comb_m:.2f}",
                "SE": "-" if np.isnan(comb_se) else f"{comb_se:.2f}",
                "SD": "-" if not sds else f"{np.nanmean(sds):.2f}",
                "Skewness": "-" if not skews else f"{np.nanmean(skews):.2f}",
                "Kurtosis": "-" if not kurts else f"{np.nanmean(kurts):.2f}",
            })
            if pv_list:
                plot_items.append((lab, pv_list[0], False))
        else:
            var = var_label_to_code[lab]
            if is_categorical(df[var]):
                mask = df[var].notna() & df["W_FSTUWT"].notna()
                cats = df.loc[mask, var]
                w = df.loc[mask, "W_FSTUWT"]
                if len(cats) == 0 or w.sum() == 0:
                    continue
                freq = cats.groupby(cats).apply(lambda s: w.loc[s.index].sum())
                total = freq.sum()
                rows = []
                for cat_val, wt in freq.items():
                    pct = 100.0 * wt / total
                    se = brr_se_percent(df, var, cat_val)
                    pretty = value_labels.get(var, {}).get(cat_val, cat_val) if var in value_labels else cat_val
                    rows.append({
                        "Variable": lab,
                        "Category": pretty,
                        "N": int((cats == cat_val).sum()),
                        "%": f"{pct:.1f}",
                        "SE": "-" if np.isnan(se) else f"{se:.1f}",
                    })
                cat_frames.append(pd.DataFrame(rows))
                plot_items.append((lab, var, True))
            else:
                mask = df[var].notna() & df["W_FSTUWT"].notna()
                x = df.loc[mask, var].astype(float).values
                w = df.loc[mask, "W_FSTUWT"].values
                se, n = brr_se_mean(df, var)
                if len(x) == 0:
                    numeric_rows.append({
                        "Variable": lab, "N": 0, "Mean": "-", "SE": "-", "SD": "-",
                        "Skewness": "-", "Kurtosis": "-",
                    })
                else:
                    numeric_rows.append({
                        "Variable": lab,
                        "N": int(n),
                        "Mean": f"{weighted_mean(x, w):.2f}",
                        "SE": "-" if np.isnan(se) else f"{se:.2f}",
                        "SD": f"{weighted_std(x, w):.2f}",
                        "Skewness": f"{weighted_skew(x, w):.2f}",
                        "Kurtosis": f"{weighted_kurtosis(x, w):.2f}",
                    })
                plot_items.append((lab, var, False))

    note = (
        "Means and percentages are weighted with W_FSTUWT. "
        "SE uses Fay BRR (k = 0.5) on 80 student replicate weights. "
        "Score-domain means combine 10 plausible values with Rubin's rules. "
        "Skewness and kurtosis are descriptive only (no SE). "
        "N is the unweighted number of cases with non-missing data."
    )

    if numeric_rows:
        num_df = pd.DataFrame(numeric_rows)
        st.markdown(
            create_apa_table_html(num_df, "Weighted descriptive statistics (continuous variables and score domains)", note),
            unsafe_allow_html=True,
        )
    if cat_frames:
        cat_df = pd.concat(cat_frames, ignore_index=True)
        st.markdown(
            create_apa_table_html(cat_df, "Weighted percentages (categorical variables)", note),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Visualisations")
    for lab, col, is_cat in plot_items:
        if not is_cat:
            mask = df[col].notna() & df["W_FSTUWT"].notna()
            x = df.loc[mask, col]
            w = df.loc[mask, "W_FSTUWT"]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
            ax1.hist(x, bins=30, weights=w, color="skyblue", edgecolor="black")
            ax1.set_title("Weighted Histogram")
            ax1.set_xlabel(lab)
            ax1.set_ylabel("Weighted Count")
            sns.boxplot(y=x, ax=ax2, width=0.5)
            ax2.set_title("Box Plot")
            ax2.set_ylabel(lab)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            mask = df[col].notna() & df["W_FSTUWT"].notna()
            cats = df.loc[mask, col]
            w = df.loc[mask, "W_FSTUWT"]
            freq = cats.groupby(cats).apply(lambda s: w.loc[s.index].sum())
            total = freq.sum()
            percentages = freq / total * 100
            if col in value_labels:
                freq.index = [value_labels[col].get(val, val) for val in freq.index]
            wrapped = [textwrap.fill(str(v), width=20) for v in freq.index]
            height = max(1.0, 2.5 + (len(freq) - 2) * 0.5)
            fig, ax = plt.subplots(figsize=(4, height))
            ax.barh(range(len(freq)), freq.values, color="skyblue", edgecolor="black")
            max_freq = max(freq.values) if len(freq) else 1
            ax.set_xlim(0, max_freq * 1.2)
            outside = max(max_freq * 0.03, 1)
            for i, (v, p) in enumerate(zip(freq.values, percentages)):
                ax.text(v + outside, i, f"{p:.1f}%", va="center", fontsize=7)
            ax.set_title(f"Weighted Frequencies for {lab}", fontsize=7)
            ax.set_xlabel("Weighted Count", fontsize=7)
            ax.set_yticks(range(len(freq)))
            ax.set_yticklabels(wrapped, fontsize=7)
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(fig)
            plt.close(fig)

# Instructions are shown under the page title.
