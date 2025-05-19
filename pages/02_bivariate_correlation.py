import streamlit as st
import pandas as pd
import numpy as np
import re
import streamlit.components.v1 as components
from scipy.stats import t, chi2_contingency

# Add logo that persists across all pages
try:
    st.logo("assets/logo.png")
except Exception as e:
    st.error(f"Failed to load logo: {e}")

# Streamlit app configuration
st.set_page_config(page_title="Bivariate Correlation - PISA Data Exploration Tool", layout="wide")

# Function to determine if a variable is categorical
def is_categorical(series, threshold=10):
    """
    Determine if a variable is categorical based on the number of unique values.
    Also checks if the variable is binary for point-biserial correlation.
    Returns: (is_categorical, is_binary)
    """
    unique_values = series.dropna().nunique()
    is_cat = unique_values <= threshold
    is_bin = unique_values == 2 if is_cat else False
    return is_cat, is_bin

# Function to compute weighted correlation coefficient (Pearson)
def weighted_pearson_correlation(x, y, w):
    try:
        if len(x) != len(y) or len(x) != len(w):
            raise ValueError("Input arrays must have the same length.")
        if not all(w >= 0):
            raise ValueError("Weights must be non-negative.")
        if len(x) <= 1:
            return np.nan, np.nan
        # Remove NaN values
        mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(w))
        x = x[mask]
        y = y[mask]
        w = w[mask]
        if len(x) <= 1:
            return np.nan, np.nan
        # Weighted mean
        w_sum = np.sum(w)
        if w_sum == 0:
            return np.nan, np.nan
        wx_mean = np.sum(w * x) / w_sum
        wy_mean = np.sum(w * y) / w_sum
        # Weighted covariance and variances
        cov_xy = np.sum(w * (x - wx_mean) * (y - wy_mean)) / w_sum
        var_x = np.sum(w * (x - wx_mean)**2) / w_sum
        var_y = np.sum(w * (y - wy_mean)**2) / w_sum
        if var_x == 0 or var_y == 0:
            return np.nan, np.nan
        # Weighted correlation
        corr = cov_xy / np.sqrt(var_x * var_y)
        # Compute p-value (approximate, using effective sample size)
        n_eff = len(x)  # Simplified; could use more sophisticated effective sample size
        if n_eff < 3:
            return corr, np.nan
        t_stat = corr * np.sqrt((n_eff - 2) / (1 - corr**2))
        p_value = 2 * (1 - t.cdf(np.abs(t_stat), df=n_eff-2))
        return corr, p_value
    except Exception as e:
        st.error(f"Error in weighted_pearson_correlation: {str(e)}")
        return np.nan, np.nan

# Function to compute weighted point-biserial correlation (continuous vs. binary categorical)
def weighted_point_biserial_correlation(x_cont, y_cat, w):
    """
    x_cont: Continuous variable
    y_cat: Binary categorical variable (assumed to be coded as 0 and 1)
    w: Weights
    """
    try:
        if len(x_cont) != len(y_cat) or len(x_cont) != len(w):
            raise ValueError("Input arrays must have the same length.")
        if not all(w >= 0):
            raise ValueError("Weights must be non-negative.")
        # Remove NaN values
        mask = ~(np.isnan(x_cont) | np.isnan(y_cat) | np.isnan(w))
        x = x_cont[mask]
        y = y_cat[mask]
        w = w[mask]
        if len(x) <= 1:
            return np.nan, np.nan
        # Ensure y is binary and coded as 0 and 1
        unique_y = np.unique(y)
        if len(unique_y) != 2:
            raise ValueError("Categorical variable must be binary for point-biserial correlation.")
        # Map to 0 and 1 if necessary
        y = (y == unique_y[1]).astype(float)
        # Weighted means
        w_sum = np.sum(w)
        if w_sum == 0:
            return np.nan, np.nan
        n_0 = np.sum(w * (1 - y))  # Sum of weights for y=0
        n_1 = np.sum(w * y)        # Sum of weights for y=1
        if n_0 == 0 or n_1 == 0:
            return np.nan, np.nan
        mean_x_0 = np.sum(w * (1 - y) * x) / n_0  # Weighted mean of x when y=0
        mean_x_1 = np.sum(w * y * x) / n_1        # Weighted mean of x when y=1
        # Weighted standard deviation of x
        wx_mean = np.sum(w * x) / w_sum
        var_x = np.sum(w * (x - wx_mean)**2) / w_sum
        std_x = np.sqrt(var_x)
        if std_x == 0:
            return np.nan, np.nan
        # Proportion of y=1
        p_1 = n_1 / w_sum
        p_0 = n_0 / w_sum
        if p_0 == 0 or p_1 == 0:
            return np.nan, np.nan
        # Point-biserial correlation
        corr = (mean_x_1 - mean_x_0) * np.sqrt(p_0 * p_1) / std_x
        # Compute p-value (approximate, using effective sample size)
        n_eff = len(x)
        if n_eff < 3:
            return corr, np.nan
        t_stat = corr * np.sqrt((n_eff - 2) / (1 - corr**2))
        p_value = 2 * (1 - t.cdf(np.abs(t_stat), df=n_eff-2))
        return corr, p_value
    except Exception as e:
        st.error(f"Error in weighted_point_biserial_correlation: {str(e)}")
        return np.nan, np.nan

# Function to compute weighted Cramér's V (categorical vs. categorical)
def weighted_cramers_v(x_cat, y_cat, w):
    """
    x_cat: First categorical variable
    y_cat: Second categorical variable
    w: Weights
    """
    try:
        if len(x_cat) != len(y_cat) or len(x_cat) != len(w):
            raise ValueError("Input arrays must have the same length.")
        if not all(w >= 0):
            raise ValueError("Weights must be non-negative.")
        # Remove NaN values
        mask = ~(np.isnan(x_cat) | np.isnan(y_cat) | np.isnan(w))
        x = x_cat[mask]
        y = y_cat[mask]
        w = w[mask]
        if len(x) <= 1:
            return np.nan, np.nan
        # Create a weighted contingency table
        categories_x = np.unique(x[~np.isnan(x)])
        categories_y = np.unique(y[~np.isnan(y)])
        if len(categories_x) < 2 or len(categories_y) < 2:
            return np.nan, np.nan
        contingency_table = np.zeros((len(categories_x), len(categories_y)))
        for i, cat_x in enumerate(categories_x):
            for j, cat_y in enumerate(categories_y):
                mask = (x == cat_x) & (y == cat_y)
                contingency_table[i, j] = np.sum(w[mask])
        # Compute chi-square statistic
        chi2, p_value, _, _ = chi2_contingency(contingency_table, correction=False)
        # Compute Cramér's V
        n = np.sum(contingency_table)  # Total weighted sample size
        if n == 0:
            return np.nan, np.nan
        k = min(len(categories_x), len(categories_y)) - 1
        if k <= 0:
            return np.nan, np.nan
        v = np.sqrt(chi2 / (n * k))
        return v, p_value
    except Exception as e:
        st.error(f"Error in weighted_cramers_v: {str(e)}")
        return np.nan, np.nan

# Function to compute BRR standard errors for correlation
def compute_brr_se_correlation(x, y, w, replicate_weight_cols, corr_data, total_work, work_done_ref, work_per_brr, placeholder, progress_placeholder, correlation_type="pearson"):
    try:
        # Initial correlation with final student weights for reference
        if correlation_type == "pearson":
            main_corr, _ = weighted_pearson_correlation(x, y, w)
        elif correlation_type == "point_biserial":
            main_corr, _ = weighted_point_biserial_correlation(x, y, w)
        elif correlation_type == "cramers_v":
            main_corr, _ = weighted_cramers_v(x, y, w)
        else:
            raise ValueError(f"Unsupported correlation type: {correlation_type}")
        if np.isnan(main_corr):
            return np.nan
        
        # Compute correlations for each replicate weight
        replicate_corrs = []
        total_weights = len(replicate_weight_cols)
        for idx, weight_col in enumerate(replicate_weight_cols):
            # Create a DataFrame with only the weights to handle NA dropping
            data = corr_data[[weight_col, 'W_FSTUWT']].copy()
            data = data.dropna()
            if len(data) < 2:
                work_done_ref[0] += work_per_brr
                continue
            
            # Subset x and y to match the non-NA indices of the weights
            indices = data.index
            pos = corr_data.index.get_indexer(indices)
            pos = pos[pos != -1]  # Remove invalid indices
            if len(pos) < 2:
                work_done_ref[0] += work_per_brr
                continue
            x_rep = x[pos]
            y_rep = y[pos]
            w_rep = data[weight_col].values
            
            # Compute correlation with replicate weights
            if correlation_type == "pearson":
                rep_corr, _ = weighted_pearson_correlation(x_rep, y_rep, w_rep)
            elif correlation_type == "point_biserial":
                rep_corr, _ = weighted_point_biserial_correlation(x_rep, y_rep, w_rep)
            elif correlation_type == "cramers_v":
                rep_corr, _ = weighted_cramers_v(x_rep, y_rep, w_rep)
            if not np.isnan(rep_corr):
                replicate_corrs.append(rep_corr)
            
            # Update progress using the placeholder
            work_done_ref[0] += work_per_brr
            progress = min(work_done_ref[0] / total_work, 1.0)
            progress_placeholder.progress(progress)
        
        if not replicate_corrs:
            return np.nan
        
        # Convert to array and compute standard error
        replicate_corrs = np.array(replicate_corrs)
        se = np.sqrt((1 / 80) * np.sum((replicate_corrs - main_corr) ** 2))
        return se
    except Exception as e:
        st.error(f"Error in compute_brr_se_correlation: {str(e)}")
        return np.nan

# Function to compute p-values using BRR standard errors
def compute_brr_p_value(corr, se, n):
    try:
        if np.isnan(se) or se == 0:
            return np.nan
        t_stat = corr / se
        p_value = 2 * (1 - t.cdf(np.abs(t_stat), df=n-2))
        return p_value
    except Exception as e:
        st.error(f"Error in compute_brr_p_value: {str(e)}")
        return np.nan

# Function to apply Rubin's rules for combining correlations across plausible values
def apply_rubins_rules_correlations(corrs_list, se_list, n):
    try:
        corrs_array = np.array(corrs_list)
        se_array = np.array(se_list)
        num_pvs = len(corrs_list)
        
        # Combined point estimate
        combined_corr = np.nanmean(corrs_array)  # Use nanmean to handle NaN values
        
        # Within-imputation variance
        within_var = np.nanmean(se_array**2)
        
        # Between-imputation variance
        between_var = np.nanvar(corrs_array, ddof=1)
        
        # Total variance
        total_var = within_var + (1 + 1/num_pvs) * between_var
        
        # Combined standard error
        combined_se = np.sqrt(total_var)
        
        # Compute p-value
        p_value = compute_brr_p_value(combined_corr, combined_se, n)
        
        return combined_corr, combined_se, p_value
    except Exception as e:
        st.error(f"Error in apply_rubins_rules_correlations: {str(e)}")
        return np.nan, np.nan, np.nan

# Function to render correlation table as HTML
def render_correlation_table(results):
    html_content = """
    <style>
    .corr-table-container {
        display: inline-block;
        overflow-x: auto;
        scrollbar-width: thin;
        min-width: 0;
        margin: 20px 0;
    }
    .corr-table-container::-webkit-scrollbar {
        height: 8px;
    }
    .corr-table-container::-webkit-scrollbar-thumb {
        background-color: #888;
        border-radius: 4px;
    }
    .corr-table {
        table-layout: fixed;
        border-collapse: collapse;
        font-size: 14px;
        margin: 0;
    }
    .corr-table th, .corr-table td {
        border: none;
        padding: 8px;
        box-sizing: border-box;
        text-align: center;
        font-weight: normal;
    }
    .corr-table th:first-child, .corr-table td:first-child {
        width: 200px !important;
        text-align: left;
        white-space: normal;
        overflow-wrap: break-word;
    }
    .corr-table th:not(:first-child), .corr-table td:not(:first-child) {
        width: 100px !important;
    }
    .corr-table tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    .corr-table-title {
        font-size: 16px;
        font-weight: bold;
        text-align: left;
        margin-bottom: 5px;
    }
    .corr-table-subtitle {
        font-size: 16px;
        font-style: italic;
        text-align: left;
        margin-bottom: 10px;
    }
    .corr-table-header {
        border-top: 1px solid #000;
        border-bottom: 1px solid #000;
    }
    .corr-table-last-row {
        border-bottom: 1px solid #000;
    }
    .corr-table-note {
        font-size: 14px;
        text-align: left;
        margin-top: 5px;
    }
    </style>
    <div class="corr-table-container">
        <div class="corr-table-title">Table 1</div>
        <div class="corr-table-subtitle">Weighted Bivariate Correlations Between Selected Variables</div>
        <table class="corr-table">
            <tr class="corr-table-header">
                <th>Variable 1</th>
                <th>Variable 2</th>
                <th>Correlation</th>
                <th>P-value</th>
                <th>Standard Error</th>
                <th>% Missing (Var1)</th>
                <th>% Missing (Var2)</th>
                <th>Type</th>
            </tr>
            {{data_rows}}
        </table>
        <div class="corr-table-note">Sample Size: Final N = {{final_size}} ({{percent_retained}}% of original N = {{original_size}} after listwise deletion)</div>
        <div class="corr-table-note">Note: Correlation types - Pearson (continuous-continuous), Point-Biserial (continuous-binary categorical), Cramér's V (categorical-categorical). For continuous vs. non-binary categorical, Pearson's correlation is used as an approximation.</div>
    </div>
    """
    data_rows = ""
    for idx, result in enumerate(results):
        # Handle potential legacy data with 9 elements
        if len(result) == 9:
            # Add a default corr_type for legacy data
            var1_label, var2_label, corr, p_value, se, miss1, miss2, final_size, original_size = result
            corr_type = "unknown"  # Default for legacy data
        elif len(result) == 10:
            var1_label, var2_label, corr, p_value, se, miss1, miss2, final_size, original_size, corr_type = result
        else:
            st.warning(f"Skipping invalid result entry with {len(result)} elements.")
            continue
        
        corr_display = f"{corr:.3f}" if not np.isnan(corr) else "-"
        p_display = "< .001" if p_value < 0.001 else f"= {p_value:.3f}" if not np.isnan(p_value) else "-"
        se_display = f"{se:.3f}" if not np.isnan(se) else "-"
        sig_display = "**" if p_value < 0.01 else "*" if p_value < 0.05 else "" if not np.isnan(p_value) else ""
        miss1_display = f"{miss1:.1f}" if miss1 is not None else "-"
        miss2_display = f"{miss2:.1f}" if miss2 is not None else "-"
        corr_type_display = {
            "pearson": "Pearson",
            "point_biserial": "Point-Biserial",
            "cramers_v": "Cramér's V",
            "pearson_approx": "Pearson (approx)",
            "unknown": "Unknown (legacy data)"
        }.get(corr_type, "-")
        row_class = "corr-table-last-row" if idx == len(results) - 1 else ""
        row = f"""
        <tr class="{row_class}">
            <th>{var1_label}</th>
            <th>{var2_label}</th>
            <td>{corr_display}{sig_display}</td>
            <td>{p_display}</td>
            <td>{se_display}</td>
            <td>{miss1_display}</td>
            <td>{miss2_display}</td>
            <td>{corr_type_display}</td>
        </tr>
        """
        data_rows += row
    
    # Use the first pair's sample size for display
    first_result = results[0]
    final_size = first_result[7]  # final_size
    original_size = first_result[8]  # original_size
    
    # Calculate percentage of original dataset retained
    percent_retained = (final_size / original_size * 100) if original_size > 0 else 0
    percent_retained_display = f"{percent_retained:.1f}"
    
    full_html = html_content.replace("{{data_rows}}", data_rows).replace("{{final_size}}", str(final_size)).replace("{{original_size}}", str(original_size)).replace("{{percent_retained}}", percent_retained_display)
    return full_html

# Access data from session state
df = st.session_state.get('df', None)
variable_labels = st.session_state.get('variable_labels', {})
value_labels = st.session_state.get('value_labels', {})
visible_columns = st.session_state.get('visible_columns', [])

# Initialize session state for this page
if 'bivariate_results' not in st.session_state:
    st.session_state.bivariate_results = None
if 'bivariate_completed' not in st.session_state:
    st.session_state.bivariate_completed = False
if 'bivariate_selected_codes' not in st.session_state:
    st.session_state.bivariate_selected_codes = None

# Check and clear session state if the structure is outdated
if 'bivariate_results' in st.session_state and st.session_state.bivariate_results:
    # Check the structure of the first result (if any)
    if len(st.session_state.bivariate_results) > 0:
        first_result = st.session_state.bivariate_results[0]
        if len(first_result) != 10:  # Expecting 10 elements now
            st.warning("Detected outdated session state data. Clearing session state to prevent errors.")
            st.session_state.bivariate_results = None
            st.session_state.bivariate_completed = False
            st.session_state.bivariate_selected_codes = None

# Streamlit UI
st.title("Bivariate Correlation Analysis")

if df is None or df.empty:
    st.warning("No data available. Please upload a dataset on the main page.")
else:
    # Detect all numeric variables (including all PVs for analysis)
    pv_pattern = re.compile(r'^PV([1-9]|10)(MATH|READ|SCIE)(\d*)$', re.IGNORECASE)
    weight_columns = ['W_FSTUWT'] + [f"W_FSTURWT{i}" for i in range(1, 81)] + [col for col in df.columns if 'W_FSCHWT' in col]
    excluded_variables = [
        'CYC', 'NATCEN', 'STRATUM', 'SUBNATIO', 'REGION', 'OECD', 'ADMINMODE',
        'LANGTEST_QQQ', 'LANGTEST_COG', 'LANGTEST_PAQ', 'OPTION_CT', 'OPTION_FL',
        'OPTION_ICTQ', 'OPTION_WBQ', 'OPTION_PQ', 'OPTION_TQ', 'OPTION_UH', 'BOOKID',
        'COBN_S', 'COBN_M', 'COBN_F', 'OCOD1', 'OCOD2', 'OCOD3',
        'ST001D01T', 'ST003D02T', 'ST003D03T',
        'EFFORT1', 'EFFORT2', 'PROGN', 'ISCEDP', 'SENWT', 'VER_DAT', 'TEST',
        'GRADE', 'UNIT', 'WVARSTRR',
        'PAREDINT', 'HISEI', 'HOMEPOS', 'BMMJ1', 'BFMJ2',
        'SCHOOLID', 'STUID', 'CNT', 'CNTRYID', 'CNTSCHID', 'CNTSTUID'
    ]
    item_pattern = re.compile(r'^(ST|FL|IC|WB|PA)\w{8}$', re.IGNORECASE)
    numeric_vars = [
        col for col in df.columns 
        if col not in weight_columns 
        and col not in excluded_variables
        and not item_pattern.match(col) 
        and not df[col].isna().all()
        and df[col].dtype in ['float64', 'int64']
    ]
    
    # Identify plausible value domains and regular numeric variables
    pv_domains = {}
    regular_numeric_vars = []
    for col in numeric_vars:
        pv_match = pv_pattern.match(col)
        if pv_match:
            pv_num = int(pv_match.group(1))
            domain = pv_match.group(2).upper()  # Normalize to uppercase
            if domain not in pv_domains:
                pv_domains[domain] = []
            pv_domains[domain].append(col)
        else:
            regular_numeric_vars.append(col)
    
    # Sort PVs within each domain to ensure PV1 to PV10 order
    for domain in pv_domains:
        pv_domains[domain].sort(key=lambda x: int(re.match(r'PV(\d+)', x, re.IGNORECASE).group(1)))
    
    # Check if PV domains have all 10 plausible values
    for domain, pv_list in pv_domains.items():
        if len(pv_list) != 10:
            st.warning(f"Domain {domain} has {len(pv_list)} plausible values instead of 10. Only {pv_list} will be used.")
    
    # Create domain options for selection
    domain_to_label = {
        'MATH': 'Mathematics score',
        'READ': 'Reading score',
        'SCIE': 'Science score'
    }
    domain_options = [domain_to_label.get(domain, domain) for domain in pv_domains.keys()]
    label_to_domain = {domain_to_label.get(domain, domain): domain for domain in pv_domains.keys()}
    
    # Prepare labels for regular numeric variables, using only visible ones for UI
    regular_numeric_vars = [col for col in regular_numeric_vars if col in visible_columns]
    var_labels = [variable_labels.get(col, col) for col in regular_numeric_vars]
    label_to_var = {variable_labels.get(col, col): col for col in regular_numeric_vars}
    unique_var_labels = []
    seen_labels = {}
    for col, label in zip(regular_numeric_vars, var_labels):
        if label in seen_labels:
            unique_label = f"{label} ({col})"
        else:
            unique_label = label
        seen_labels[label] = True
        unique_var_labels.append(unique_label)
        label_to_var[unique_label] = col
    
    if len(unique_var_labels) + len(domain_options) < 2:
        st.warning("At least two numeric variables or domains are required for bivariate correlation analysis.")
    else:
        # Select Variables (default to MATH domain and ESCS)
        st.write("Select variables for bivariate correlation (default: Mathematics score, ESCS):")
        all_var_options = domain_options + unique_var_labels
        default_vars = []
        if "Mathematics score" in all_var_options:
            default_vars.append("Mathematics score")
        escs_label = variable_labels.get("ESCS", "ESCS")
        if escs_label in seen_labels and escs_label in all_var_options:
            default_vars.append(escs_label)
        elif all_var_options and len(all_var_options) > 1:
            default_vars.append(all_var_options[1 if "Mathematics score" in all_var_options else 0])
        
        if len(default_vars) < 2 and len(all_var_options) >= 2:
            default_vars = all_var_options[:2]
        
        selected_var_labels = st.multiselect(
            "Variables",
            all_var_options,
            default=default_vars,
            key="bivariate_vars"
        )
        
        # Split selected variables into domains and regular variables
        selected_domains = []
        selected_vars = []
        selected_codes = []
        all_vars = []
        for label in selected_var_labels:
            if label in label_to_domain:
                domain = label_to_domain[label]
                selected_domains.append(domain)
                selected_codes.append(domain)
                all_vars.append(pv_domains[domain])
            else:
                selected_vars.append(label_to_var[label])
                selected_codes.append(label_to_var[label])
                all_vars.append([label_to_var[label]])
        
        run_analysis = st.button("Run Analysis", key="run_bivariate")
        
        if run_analysis and (selected_vars or selected_domains) and len(selected_var_labels) >= 2:
            try:
                if 'W_FSTUWT' not in df.columns:
                    st.error("Final student weight (W_FSTUWT) not found in the dataset.")
                else:
                    # Check for replicate weights availability
                    replicate_weight_cols = [f"W_FSTURWT{i}" for i in range(1, 81)]
                    missing_weights = [col for col in replicate_weight_cols if col not in df.columns]
                    use_brr = len(missing_weights) == 0
                    
                    # Compute the maximum number of plausible values
                    max_pvs = max([len(vars) for vars in all_vars])
                    
                    # Calculate total work for the progress bar
                    num_pairs = (len(all_vars) * (len(all_vars) - 1)) // 2
                    pv_iterations = num_pairs * max_pvs  # Total PV iterations
                    brr_iterations_per_pv = len(replicate_weight_cols) if use_brr else 0
                    total_brr_iterations = pv_iterations * brr_iterations_per_pv
                    total_work = pv_iterations + total_brr_iterations
                    work_done = 0
                    work_per_pv = 1  # Each PV iteration is 1 unit of work
                    work_per_brr = 1 / len(replicate_weight_cols) if use_brr else 0  # Each BRR iteration is a fraction of a PV unit
                    
                    # Single main progress bar placeholder
                    progress_placeholder = st.empty()
                    progress_placeholder.progress(0.0)
                    
                    # Single placeholder for messages
                    placeholder = st.empty()
                    
                    # Compute correlations for each pair of variables
                    table_results = []
                    completed_pv_iterations = 0
                    for i in range(len(all_vars)):
                        for j in range(i + 1, len(all_vars)):
                            var1_list = all_vars[i]
                            var2_list = all_vars[j]
                            var1_is_pv = len(var1_list) > 1
                            var2_is_pv = len(var2_list) > 1
                            num_pvs = max(len(var1_list), len(var2_list))
                            
                            all_corrs = []
                            all_p_values = []
                            all_se = []
                            correlation_type = None
                            
                            # Determine variable types
                            var1 = var1_list[0]
                            var2 = var2_list[0]
                            # For PV domains, treat as continuous
                            var1_is_cat, var1_is_binary = (False, False) if var1_is_pv else is_categorical(df[var1])
                            var2_is_cat, var2_is_binary = (False, False) if var2_is_pv else is_categorical(df[var2])
                            
                            # Determine correlation type
                            if not var1_is_cat and not var2_is_cat:
                                correlation_type = "pearson"
                            elif var1_is_cat and var2_is_cat:
                                correlation_type = "cramers_v"
                            elif not var1_is_cat and var2_is_cat:
                                if var2_is_binary:
                                    correlation_type = "point_biserial"
                                else:
                                    correlation_type = "pearson_approx"
                                    placeholder.write(f"Using Pearson's correlation as an approximation for {selected_var_labels[i]} (continuous) vs. {selected_var_labels[j]} (non-binary categorical).")
                            elif var1_is_cat and not var2_is_cat:
                                if var1_is_binary:
                                    correlation_type = "point_biserial"
                                else:
                                    correlation_type = "pearson_approx"
                                    placeholder.write(f"Using Pearson's correlation as an approximation for {selected_var_labels[i]} (non-binary categorical) vs. {selected_var_labels[j]} (continuous).")
                            
                            # Calculate missing percentages once per variable pair
                            pair_data = df[[var1, var2, 'W_FSTUWT']].copy()
                            total_rows = len(pair_data)
                            pair_data = pair_data.dropna()
                            final_size = len(pair_data)
                            miss1 = (total_rows - len(df[var1].dropna())) / total_rows * 100
                            miss2 = (total_rows - len(df[var2].dropna())) / total_rows * 100
                            
                            for pv_idx in range(num_pvs):
                                var1 = var1_list[pv_idx % len(var1_list)]
                                var2 = var2_list[pv_idx % len(var2_list)]
                                placeholder.write(f"Processing correlation between {var1} and {var2} (Type: {correlation_type})...")
                                
                                # Create a subset DataFrame for just this pair
                                pair_columns = ['W_FSTUWT']
                                if use_brr:
                                    pair_columns += replicate_weight_cols
                                pair_columns.append(var1)
                                pair_columns.append(var2)
                                pair_data = df[pair_columns].copy()
                                total_rows_pv = len(pair_data)
                                pair_data = pair_data.dropna()
                                final_size_pv = len(pair_data)
                                
                                # Debug insufficient data
                                if final_size_pv < 2:
                                    placeholder.write(f"Insufficient non-missing data for correlation between {var1} and {var2}. Before dropping NA: {total_rows_pv} rows, after dropping NA: {final_size_pv} rows.")
                                    all_corrs.append(np.nan)
                                    all_p_values.append(np.nan)
                                    all_se.append(np.nan)
                                    # Increment work_done for the PV iteration and its BRR iterations
                                    work_done += work_per_pv
                                    if use_brr:
                                        work_done += brr_iterations_per_pv * work_per_brr
                                    completed_pv_iterations += 1
                                    progress = min(work_done / total_work, 1.0)
                                    progress_placeholder.progress(progress)
                                    continue
                                
                                x = pair_data[var1].values
                                y = pair_data[var2].values
                                w = pair_data['W_FSTUWT'].values
                                
                                # Compute correlation based on type
                                if correlation_type == "pearson" or correlation_type == "pearson_approx":
                                    corr, p_value = weighted_pearson_correlation(x, y, w)
                                elif correlation_type == "point_biserial":
                                    # Determine which variable is continuous
                                    if not var1_is_cat:
                                        corr, p_value = weighted_point_biserial_correlation(x, y, w)
                                    else:
                                        corr, p_value = weighted_point_biserial_correlation(y, x, w)
                                elif correlation_type == "cramers_v":
                                    corr, p_value = weighted_cramers_v(x, y, w)
                                
                                if use_brr:
                                    placeholder.write("Calculating standard error using replicate weights...")
                                    # Pass work_done as a list to allow modification in the function
                                    work_done_ref = [work_done]
                                    se = compute_brr_se_correlation(x, y, w, replicate_weight_cols, pair_data, total_work, work_done_ref, work_per_brr, placeholder, progress_placeholder, correlation_type)
                                    work_done = work_done_ref[0]
                                    p_value = compute_brr_p_value(corr, se, len(pair_data))
                                else:
                                    se = np.nan
                                    work_done += work_per_pv
                                    completed_pv_iterations += 1
                                    progress = min(work_done / total_work, 1.0)
                                    progress_placeholder.progress(progress)
                                
                                all_corrs.append(corr)
                                all_p_values.append(p_value)
                                all_se.append(se)
                                
                                if not use_brr:  # Progress already updated in compute_brr_se_correlation if use_brr is True
                                    completed_pv_iterations += 1
                                    progress = min(work_done / total_work, 1.0)
                                    progress_placeholder.progress(progress)
                        
                            # Combine results
                            if num_pvs == 1:
                                placeholder.write(f"Combining results for {selected_var_labels[i]} and {selected_var_labels[j]} (single iteration, no plausible values)...")
                                combined_corr = all_corrs[0]
                                combined_p_value = all_p_values[0]
                                combined_se = all_se[0]
                            else:
                                placeholder.write(f"Combining results for {selected_var_labels[i]} and {selected_var_labels[j]}...")
                                combined_corr, combined_se, combined_p_value = apply_rubins_rules_correlations(all_corrs, all_se, len(pair_data))
                            
                            # Store results as a single tuple, including correlation_type
                            table_results.append((
                                selected_var_labels[i],
                                selected_var_labels[j],
                                combined_corr,
                                combined_p_value,
                                combined_se,
                                miss1,
                                miss2,
                                final_size,
                                total_rows,
                                correlation_type
                            ))
                    
                    # Ensure progress reaches 100%
                    progress_placeholder.progress(1.0)
                    
                    # Store results and selected_codes in session state
                    st.session_state.bivariate_results = table_results
                    st.session_state.bivariate_selected_codes = selected_codes
                    st.session_state.bivariate_completed = True
                    
                    # Render the correlation table
                    placeholder.write("Rendering correlation table...")
                    table_html = render_correlation_table(table_results)
                    components.html(table_html, height=400, scrolling=True)
                    placeholder.write("Bivariate correlation analysis completed.")
            except Exception as e:
                placeholder.empty()
                st.error(f"Error computing bivariate correlations: {str(e)}")
                st.session_state.bivariate_completed = False
                st.session_state.bivariate_results = None
                st.session_state.bivariate_selected_codes = None
        elif st.session_state.bivariate_results and st.session_state.bivariate_completed:
            if selected_var_labels:
                table_results = st.session_state.bivariate_results
                selected_codes = st.session_state.bivariate_selected_codes
                table_html = render_correlation_table(table_results)
                components.html(table_html, height=400, scrolling=True)
                st.write("Bivariate correlation analysis completed.")
            else:
                st.write("Please select at least two variables to compute bivariate correlations.")
        else:
            st.write("Please select at least two variables and click 'Run Analysis' to compute bivariate correlations.")

# Instructions section
st.header("Instructions")
st.markdown("""
- **Select Variables**: Choose two or more variables (domains or numeric variables) from the dropdown menus. Plausible value domains (e.g., Mathematics score, Reading score) will use all 10 plausible values for analysis.
- **Run Analysis**: Click "Run Analysis" to perform the weighted bivariate correlation analysis. Analyses involving plausible values will be combined using Rubin's rules.
- **View Results**: Results are displayed in an APA-style table. The correlation type is indicated for each pair: Pearson (continuous-continuous), Point-Biserial (continuous-binary categorical), Cramér's V (categorical-categorical). For continuous vs. non-binary categorical pairs, Pearson's correlation is used as an approximation.
- **Navigate**: Use the sidebar to switch between different analysis types or return to the main page to upload a new dataset.
""")