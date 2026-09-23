import streamlit as st

try:
    st.logo("assets/logo.png")
except Exception:
    pass

st.set_page_config(page_title="About - PISA Data Exploration Tool", layout="wide")

st.title("About this tool")
label = st.session_state.get("dataset_label")
if label:
    st.info(f"Dataset currently loaded: {label}")

st.header("What is Streamlit?")
st.markdown(
    """
Streamlit is a way to turn Python analysis scripts into a **website you use in a
browser**. There is no SPSS, RStudio or command line for the person exploring
the data. You open a link, upload a country `.sav`, and click through pages
(descriptives, subgroups, regression, and so on).

The app runs on a server (here, Streamlit Community Cloud). Your file is held
in that session’s memory while you work; it is not written into the public
GitHub repository. Close the tab or upload another file and that session’s data
go away.

You do not need to install Python to *use* the shared link. Python is only
needed if you run a copy on your own PC.
"""
)

st.markdown(
    """
This app is a **country-file exploration tool** for PISA student extracts
(and student files merged with school variables). It is for colleagues to inspect
weighted associations. It is **not** a substitute for OECD publications or the
IEA IDB Analyzer in a paper’s methods statement.
"""
)

st.header("Design-based methods")
st.markdown(
    """
- **Weight.** Every estimate uses the final student weight `W_FSTUWT`.
  Analyses will not run if the 80 student replicate weights `W_FSTURWT1`–`W_FSTURWT80` are missing.
- **Variance.** Fay balanced repeated replication with *k* = 0.5 and *G* = 80:
  SE = √[(1/20) Σ(θ<sub>g</sub> − θ)²]. This is the official PISA factor (not 1/80).
- **Plausible values.** Mathematics, reading and science are entered as **domains**.
  All 10 PVs are analysed and combined with Rubin’s rules. Individual PV columns
  are not offered as separate outcomes.
- **Missing data.** Listwise deletion within each analysis. *N* is the unweighted
  count with complete data for that specification.
- **Stars.** \\* *p* < .01, \\*\\* *p* < .001. *p* < .05 is not starred, because
  PISA samples are large.
"""
)

st.header("What each page does")
st.markdown(
    """
- **Descriptive statistics** — weighted mean and Fay-BRR SE (Rubin if a score domain);
  SD, skewness and kurtosis without SEs. Categorical variables: weighted % and BRR SE.
- **Subgroup tables** — means or percentages by gender, school type, other short
  categoricals, or country-specific weighted ESCS tertiles (cuts fixed across replicates).
  Gaps versus a reference group have a BRR SE on the difference. Score domains can also
  be shown as % below Level 2 and % at Levels 5–6.
- **Correlations** — weighted Pearson, point-biserial or Cramér’s V; BRR SEs;
  Rubin when a domain is included.
- **Linear regression** — weighted OLS. Coefficient SEs and *p*-values use Fay BRR
  (+ Rubin if *Y* is a domain). *R*² and max VIF come from the single WLS fit and
  are **not** design-based. Residual plots are informal. Model *F*, Anderson–Darling
  and Breusch–Pagan are not printed.
- **Regression with interactions** — same engine, plus product terms among
  **observed** variables only (not score × score). Continuous predictors can be centred.
  Simple slopes at −1 SD / mean / +1 SD (or the two codes of a binary) use BRR + Rubin.
- **Percentiles and proficiency levels** — P5–P95 and P90−P10 with BRR + Rubin SEs.
  Level shares use official PISA 2025 cut-scores. Level 2 is the OECD baseline.
"""
)

st.header("Check against published United Kingdom 2025 figures")
st.markdown(
    """
On the UK student file the tool reproduced OECD Education GPS country means:

- Science **511.36** (OECD **511**)
- Mathematics **487.57** (OECD **488**)
- Reading **494.23** (OECD **494**)

Science by gender: girls 505.4, boys 516.9, gap **11.5** (OECD boys–girls gap **12**).

These are **United Kingdom** figures, not England-only (England science is 516).
"""
)

st.header("Intentionally not included")
st.markdown(
    """
- Mediation or other causal path models (one PISA cycle cannot time-order
  predictor, mediator and outcome).
- Design-based model *F* or residual tests.
- Two score domains in one regression (the PV loop is built for one domain).
- TIMSS-style jackknife or school-level replicate systems.
- Official OECD recode dictionaries.
"""
)

st.header("How this differs from IDB Analyzer")
st.markdown(
    """
The sampling estimator for means, percentages, correlations and main-effects
regression is the **same family** as IEA IDB Analyzer on a PISA student file.
IDB remains the institutional standard for publications.

This tool adds interactions with design-based simple slopes, BRR SEs on
subgroup gaps, and 2025 proficiency cuts in a browser. Treat it as
**design-aware exploration**, not certified OECD output.
"""
)

st.header("PisaSplitter (companion desktop tool)")
st.markdown(
    """
PisaSplitter is a Windows utility that prepares files for this app. It does
**not** analyse data, change weights, or alter plausible values.

- **Split** an official PISA SPSS `.sav` by `CNTRYID` into one country file
  (cycles 2015, 2018, 2022, 2025).
- **Country names.** PISA 2025 `CNTRYID` is ISO numeric. Where value labels
  do not map cleanly, names come from an ISO list (e.g. 826 = United Kingdom,
  784 = UAE).
- **Merge** a country student file with the matching school file on school ID.
  School variables are copied onto every student in that school. Student
  weights and replicates are unchanged.
- **Names.** Extracts look like `United_Kingdom_PISA2025.sav`. Merged files
  include the country, e.g. `Australia_Merged_PISA2025.sav`.

Typical path: download the international PUF → split/merge in PisaSplitter →
upload the country `.sav` here. After a merge, school variables are constant
within school; analyses of student outcomes still use **student** weights,
which is what this app does.

The splitter is a convenience extract. Keep the original OECD international
files. New 2025 participants missing from the ISO list will show as a numeric
code until that list is updated.
"""
)

st.caption(
    "Methods note for colleagues and statistical review. "
    "Upload data on the Data Upload page in the sidebar."
)
