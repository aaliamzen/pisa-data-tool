import streamlit as st
import pandas as pd
import numpy as np
import re
import streamlit.components.v1 as components
from scipy.stats import t, chi2_contingency
import matplotlib.pyplot as plt
from io import BytesIO
import textwrap
import itertools

# Script version identifier
SCRIPT_VERSION = "2025-05-19-v9"

# Add logo that persists across all pages
try:
    st.logo("assets/logo.png")
except Exception as e:
    st.error(f"Failed to load logo: {e}")

# Streamlit app configuration
st.set_page_config(page_title="Correlation Matrix - PISA Data Exploration Tool", layout="wide")

# Function to determine if a variable is categorical
def is_categorical(series, threshold=50):
    """
    Determine if a variable is categorical based on the number of unique values.
    Also checks if the variable is binary for point-biserial correlation.
    Returns: (is_categorical, is_binary, unique_values_count)
    """
    unique_values = pd.Series(series).dropna().nunique()
    is_cat = unique_values <= threshold
    is_bin = unique_values == 2 if is_cat else False
    return is_cat, is_bin, unique_values

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
            raise ValueError(f"Categorical variable must be binary for point-biserial correlation. Found {len(unique_y)} unique values: {unique_y}")
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
def compute_brr_se_correlation(cont_var, cat_var, w, replicate_weight_cols, corr_data, total_work, work_done_ref, work_per_brr, placeholder, progress_placeholder, correlation_type="pearson", cat_var_index=None):
    try:
        placeholder.write(f"Computing BRR standard error with correlation type: {correlation_type}")
        # Initial correlation with final student weights for reference
        if correlation_type in ["pearson", "pearson_approx"]:
            main_corr, _ = weighted_pearson_correlation(cont_var, cat_var, w)
        elif correlation_type == "point_biserial":
            main_corr, _ = weighted_point_biserial_correlation(cont_var, cat_var, w)
        elif correlation_type == "cramers_v":
            main_corr, _ = weighted_cramers_v(cont_var, cat_var, w)
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
            
            # Subset cont_var and cat_var to match the non-NA indices of the weights
            indices = data.index
            pos = corr_data.index.get_indexer(indices)
            pos = pos[pos != -1]  # Remove invalid indices
            if len(pos) < 2:
                work_done_ref[0] += work_per_brr
                continue
            cont_var_rep = cont_var[pos]
            cat_var_rep = cat_var[pos]
            w_rep = data[weight_col].values
            
            # Compute correlation with replicate weights
            if correlation_type in ["pearson", "pearson_approx"]:
                rep_corr, _ = weighted_pearson_correlation(cont_var_rep, cat_var_rep, w_rep)
            elif correlation_type == "point_biserial":
                rep_corr, _ = weighted_point_biserial_correlation(cont_var_rep, cat_var_rep, w_rep)
            elif correlation_type == "cramers_v":
                rep_corr, _ = weighted_cramers_v(cont_var_rep, cat_var_rep, w_rep)
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

def create_weighted_scatter(df, var1, var2, weights, var1_label, var2_label, corr, corr_type):
    """
    Create a weighted scatter plot with a regression line.
    
    Parameters:
    - df: pandas DataFrame containing the data.
    - var1, var2: Column names for x and y variables.
    - weights: pandas Series or array of weights for each data point.
    - var1_label, var2_label: Labels for x and y axes.
    - corr: Correlation coefficient to display.
    - corr_type: Type of correlation (e.g., "Pearson", "Point-Biserial", "Cramér's V").
    
    Returns:
    - fig: matplotlib Figure object.
    """
    # Remove rows with NaN in var1, var2, or weights
    valid_mask = (~df[var1].isna()) & (~df[var2].isna()) & (~weights.isna())
    x = df[var1][valid_mask].values
    y = df[var2][valid_mask].values
    w = weights[valid_mask].values

    # Check if there’s enough data to plot
    if len(x) < 2 or len(y) < 2 or len(w) < 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Insufficient data to plot (too many NaN values)", 
                ha='center', va='center', fontsize=10, color='red')
        ax.set_xlabel(var1_label, fontsize=8)
        ax.set_ylabel(var2_label, fontsize=8)
        plt.tight_layout()
        return fig

    # Create figure and axis objects with slightly larger size
    fig, ax = plt.subplots(figsize=(6, 4))

    # Normalize weights for visualization (between 10 and 100 for point sizes)
    w_min, w_max = w.min(), w.max()
    if w_max == w_min:  # Avoid division by zero
        w_norm = np.full_like(w, 50)  # Default size if all weights are the same
    else:
        w_norm = 10 + (90 * (w - w_min) / (w_max - w_min))

    # Compute weighted regression line
    x_mean = np.average(x, weights=w)
    y_mean = np.average(y, weights=w)
    cov_xy = np.sum(w * (x - x_mean) * (y - y_mean)) / np.sum(w)
    var_x = np.sum(w * (x - x_mean)**2) / np.sum(w)
    slope = cov_xy / var_x if var_x != 0 else 0
    intercept = y_mean - slope * x_mean

    # Create scatter plot with smaller points
    scatter = ax.scatter(x, y, s=w_norm, alpha=0.5, c=w, cmap='viridis', edgecolor='none')

    # Add weighted regression line (only for Pearson or Point-Biserial correlations)
    if corr_type in ["Pearson", "Point-Biserial", "Pearson (approx)"]:
        x_range = np.array([x.min(), x.max()])
        y_range = slope * x_range + intercept
        ax.plot(x_range, y_range, "r--", alpha=0.8, label=f'Regression (r={corr:.2f}, {corr_type})')
    else:
        # For Cramér's V, just show the correlation value
        ax.plot([], [], ' ', label=f'Corr ({corr_type}) = {corr:.2f}')

    # Add gridlines for better readability
    ax.grid(True, linestyle='--', alpha=0.7)

    # Wrap the axis labels to a maximum width of 50 characters
    wrapped_xlabel = textwrap.fill(var1_label, width=50)
    wrapped_ylabel = textwrap.fill(var2_label, width=50)

    # Set labels with smaller fonts
    ax.set_xlabel(wrapped_xlabel, fontsize=8)
    ax.set_ylabel(wrapped_ylabel, fontsize=8)
    
    # Add legend with smaller font
    ax.legend(fontsize=7)

    # Add colorbar with smaller label
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Weight', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    return fig

# Function to render correlation matrix as HTML table with superscript letters
def render_correlation_matrix(selected_labels, corr_matrix, p_matrix, corr_types, type_to_letter):
    html_content = """
    <style>
    table, th, td {  
            font-weight: normal !important;  
    }
    
    .corr-matrix-container {
        display: inline-block;
        overflow-x: auto;
        scrollbar-width: thin;
        min-width: 0;
        margin: 20px 0;
    }
    .corr-matrix-container::-webkit-scrollbar {
        height: 8px;
    }
    .corr-matrix-container::-webkit-scrollbar-thumb {
        background-color: #888;
        border-radius: 4px;
    }
    .corr-matrix {
        table-layout: fixed;
        border-collapse: collapse;
        font-size: 14px;
        margin: 0;
    }
    .corr-matrix th, .corr-matrix td {
        border: none;
        padding: 8px;
        box-sizing: border-box;
        text-align: center;
    }
    .corr-matrix th:first-child, .corr-matrix td:first-child {
        width: 200px !important;
        text-align: left;
        white-space: normal;
        overflow-wrap: break-word;
    }
    .corr-matrix th:not(:first-child), .corr-matrix td:not(:first-child) {
        width: 100px !important;
    }
    .corr-matrix tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    .corr-matrix-title {
        font-size: 16px;
        font-weight: bold;
        text-align: left;
        margin-bottom: 5px;
    }
    .corr-matrix-subtitle {
        font-size: 16px;
        font-style: italic;
        text-align: left;
        margin-bottom: 10px;
    }
    .corr-matrix-header {
        border-top: 1px solid #000;
        border-bottom: 1px solid #000;
    }
    .corr-matrix-last-row {
        border-bottom: 1px solid #000;
    }
    .corr-matrix-note {
        font-size: 14px;
        text-align: left;
        margin-top: 5px;
    }
    </style>
    <div class="corr-matrix-container">
        <div class="corr-matrix-title">Correlation Matrix</div>
        <div class="corr-matrix-subtitle">Weighted Correlations Between Selected Variables</div>
        <table class="corr-matrix">
            <tr class="corr-matrix-header">
                <th></th>
                {{header_row}}
            </tr>
            {{data_rows}}
        </table>
        <div class="corr-matrix-note">{{correlation_types_note}} Significance: *p < 0.01, **p < 0.001</div>
    </div>
    """
    # Generate header row
    header_row = ""
    for label in selected_labels:
        header_row += f"<th>{label}</th>"
    
    # Generate data rows
    data_rows = ""
    for i in range(len(selected_labels)):
        row_class = "corr-matrix-last-row" if i == len(selected_labels) - 1 else ""
        row = f'<tr class="{row_class}"><th>{selected_labels[i]}</th>'
        for j in range(len(selected_labels)):
            if i == j:
                cell_content = "1.00"
            else:
                corr = corr_matrix[i, j]
                p_value = p_matrix[i, j]
                pair_key = (min(i, j), max(i, j))
                corr_type = corr_types.get(pair_key, "unknown")
                letter = type_to_letter.get(corr_type, "")
                corr_display = f"{corr:.2f}" if not np.isnan(corr) else "-"
                if not np.isnan(p_value):
                    if p_value < 0.001:
                        sig_display = "**"  # Mark p < 0.001 with **
                    elif p_value < 0.01:
                        sig_display = "*"  # Mark p < 0.01 with *
                    else:
                        sig_display = ""  # No marker for p >= 0.01
                else:
                    sig_display = ""
                if letter:
                    cell_content = f"{corr_display}{sig_display}<sup>{letter}</sup>"
                else:
                    cell_content = f"{corr_display}{sig_display}"
            row += f"<td>{cell_content}</td>"
        row += "</tr>"
        data_rows += row
    
    # Generate the correlation types note
    corr_type_descriptions = {
        "pearson": "Pearson (continuous-continuous)",
        "point_biserial": "Point-Biserial (continuous-binary categorical)",
        "cramers_v": "Cramér's V (categorical-categorical)",
        "pearson_approx": "Pearson (approx, continuous-non-binary categorical)"
    }
    note_parts = []
    for corr_type, letter in type_to_letter.items():
        description = corr_type_descriptions.get(corr_type, "Unknown")
        note_parts.append(f"<sup>{letter}</sup>: {description}")
    correlation_types_note = "Correlation types: " + "; ".join(note_parts) + "."
    
    full_html = html_content.replace("{{header_row}}", header_row).replace("{{data_rows}}", data_rows).replace("{{correlation_types_note}}", correlation_types_note)
    return full_html

# Access data from session state
df = st.session_state.get('df', None)
variable_labels = st.session_state.get('variable_labels', {})
value_labels = st.session_state.get('value_labels', {})
visible_columns = st.session_state.get('visible_columns', [])

# Initialize session state for this page
if 'correlation_matrix_results' not in st.session_state:
    st.session_state.correlation_matrix_results = None
if 'correlation_matrix_completed' not in st.session_state:
    st.session_state.correlation_matrix_completed = False
if 'correlation_matrix_used_brr' not in st.session_state:
    st.session_state.correlation_matrix_used_brr = False

# Streamlit UI
st.title("Correlation Matrix Analysis")

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
        st.warning("At least two numeric variables or domains are required for correlation matrix analysis.")
    else:
        # Select Variables (default to MATH domain only)
        st.write("Select variables for correlation matrix (default: Mathematics score):")
        all_var_options = domain_options + unique_var_labels
        default_vars = []
        if "Mathematics score" in all_var_options:
            default_vars.append("Mathematics score")
        
        selected_var_labels = st.multiselect(
            "Variables",
            all_var_options,
            default=default_vars,
            key="matrix_vars"
        )
        
        # Split selected variables into domains and regular variables
        selected_domains = []
        selected_vars = []
        selected_labels = []
        selected_codes = []
        all_vars = []
        variable_types = {}  # Store whether each variable is categorical and binary
        
        for label in selected_var_labels:
            if label in label_to_domain:
                domain = label_to_domain[label]
                selected_domains.append(domain)
                selected_labels.append(domain_to_label.get(domain, domain))
                selected_codes.append(domain)
                all_vars.append(pv_domains[domain])
                # PV domains are treated as continuous
                # Use the first plausible value to determine unique values
                if pv_domains[domain]:  # Ensure the domain has at least one PV
                    first_pv = pv_domains[domain][0]  # e.g., PV1MATH
                    is_cat, is_bin, unique_count = is_categorical(df[first_pv])
                else:
                    is_cat, is_bin, unique_count = False, False, 0
                variable_types[domain] = (is_cat, is_bin, unique_count)
            else:
                var_code = label_to_var[label]
                selected_vars.append(var_code)
                selected_labels.append(label)
                selected_codes.append(var_code)
                all_vars.append([var_code])
                # Determine if the variable is categorical
                is_cat, is_bin, unique_count = is_categorical(df[var_code])
                variable_types[var_code] = (is_cat, is_bin, unique_count)
        
        run_analysis = st.button("Run Analysis", key="run_correlation_matrix")
        
        if run_analysis and (selected_vars or selected_domains) and len(selected_var_labels) >= 2:
            try:
                if 'W_FSTUWT' not in df.columns:
                    st.error("Final student weight (W_FSTUWT) not found in the dataset.")
                else:
                    # Check for replicate weights availability
                    replicate_weight_cols = [f"W_FSTURWT{i}" for i in range(1, 81)]
                    missing_weights = [col for col in replicate_weight_cols if col not in df.columns]
                    st.session_state.correlation_matrix_used_brr = len(missing_weights) == 0  # Use BRR if no replicate weights are missing
                    
                    # Initialize correlation and p-value matrices
                    n_vars = len(selected_var_labels)
                    corr_matrix = np.ones((n_vars, n_vars))
                    p_matrix = np.zeros((n_vars, n_vars))
                    corr_types = {}  # Store correlation type for each pair
                    type_to_letter = {}  # Map correlation types to superscript letters
                    letter_counter = 0  # To assign letters a, b, c, etc.
                    letters = 'abcdefghijklmnopqrstuvwxyz'
                    
                    # Compute the maximum number of plausible values
                    max_pvs = max([len(vars) for vars in all_vars])
                    
                    # Calculate total work for the progress bar
                    num_pairs = (n_vars * (n_vars - 1)) // 2
                    pv_iterations = num_pairs * max_pvs  # Total PV iterations
                    brr_iterations_per_pv = len(replicate_weight_cols) if st.session_state.correlation_matrix_used_brr else 0
                    total_brr_iterations = pv_iterations * brr_iterations_per_pv
                    total_work = pv_iterations + total_brr_iterations
                    work_done = 0
                    work_per_pv = 1  # Each PV iteration is 1 unit of work
                    work_per_brr = 1 / len(replicate_weight_cols) if st.session_state.correlation_matrix_used_brr else 0  # Each BRR iteration is a fraction of a PV unit
                    
                    # Single main progress bar placeholder
                    progress_placeholder = st.empty()
                    progress_placeholder.progress(0.0)
                    
                    # Single placeholder for messages
                    placeholder = st.empty()
                    
                    # Compute correlations for each pair of variables
                    completed_pv_iterations = 0
                    for i in range(n_vars):
                        for j in range(i + 1, n_vars):
                            var1_list = all_vars[i]
                            var2_list = all_vars[j]
                            var1_is_pv = len(var1_list) > 1
                            var2_is_pv = len(var2_list) > 1
                            num_pvs = max(len(var1_list), len(var2_list))
                            
                            all_corrs = []
                            all_p_values = []  # Store p-values directly
                            all_se = []
                            base_correlation_type = None  # Store the intended correlation type before PV loop
                            cat_var_index = None  # Store which variable is categorical (0 for var1, 1 for var2)
                            
                            # Determine variable types
                            var1_code = selected_codes[i]
                            var2_code = selected_codes[j]
                            var1_is_cat, var1_is_binary, var1_unique_count = variable_types[var1_code]
                            var2_is_cat, var2_is_binary, var2_unique_count = variable_types[var2_code]
                            
                            # Determine base correlation type (before PV loop)
                            if not var1_is_cat and not var2_is_cat:
                                base_correlation_type = "pearson"
                                cat_var_index = None  # No categorical variable
                            elif var1_is_cat and var2_is_cat:
                                base_correlation_type = "cramers_v"
                                cat_var_index = 0  # Both are categorical, default to var1 for consistency
                            elif not var1_is_cat and var2_is_cat:
                                if var2_is_binary:
                                    base_correlation_type = "point_biserial"
                                    cat_var_index = 1  # var2 is categorical
                                else:
                                    base_correlation_type = "pearson_approx"
                                    cat_var_index = 1
                                    placeholder.write(f"Using Pearson's correlation as an approximation for {selected_labels[i]} (continuous) vs. {selected_labels[j]} (non-binary categorical).")
                            elif var1_is_cat and not var2_is_cat:
                                if var1_is_binary:
                                    base_correlation_type = "point_biserial"
                                    cat_var_index = 0  # var1 is categorical
                                else:
                                    base_correlation_type = "pearson_approx"
                                    cat_var_index = 0
                                    placeholder.write(f"Using Pearson's correlation as an approximation for {selected_labels[i]} (non-binary categorical) vs. {selected_labels[j]} (continuous).")
                            
                            # Assign a superscript letter to this correlation type if not already assigned
                            if base_correlation_type not in type_to_letter:
                                if letter_counter < len(letters):
                                    type_to_letter[base_correlation_type] = letters[letter_counter]
                                    letter_counter += 1
                            
                            # Store correlation type for this pair
                            corr_types[(i, j)] = base_correlation_type
                            corr_types[(j, i)] = base_correlation_type
                            
                            for pv_idx in range(num_pvs):
                                var1 = var1_list[pv_idx % len(var1_list)]
                                var2 = var2_list[pv_idx % len(var2_list)]
                                placeholder.write(f"Processing correlation between {var1} and {var2}...")
                                
                                # Create a subset DataFrame for just this pair to ensure consistent NA handling
                                pair_columns = ['W_FSTUWT']
                                if st.session_state.correlation_matrix_used_brr:
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
                                    if st.session_state.correlation_matrix_used_brr:
                                        work_done += brr_iterations_per_pv * work_per_brr
                                    completed_pv_iterations += 1
                                    progress = min(work_done / total_work, 0.7)
                                    progress_placeholder.progress(progress)
                                    continue
                                
                                x = pair_data[var1].values
                                y = pair_data[var2].values
                                w = pair_data['W_FSTUWT'].values
                                
                                # Recompute variable types after subsetting to ensure correct classification
                                correlation_type = base_correlation_type
                                corr, p_value = None, None
                                if correlation_type == "point_biserial":
                                    # Determine which variable is categorical based on cat_var_index
                                    cat_var = x if cat_var_index == 0 else y
                                    cont_var = y if cat_var_index == 0 else x
                                    cat_var_name = var1 if cat_var_index == 0 else var2
                                    cont_var_name = var2 if cat_var_index == 0 else var1
                                    cat_is_cat, cat_is_binary, cat_unique_count = is_categorical(cat_var)
                                    if cat_is_binary:
                                        # Safety check: Ensure arguments are correct
                                        cont_unique_count = pd.Series(cont_var).dropna().nunique()
                                        cat_unique_count_check = pd.Series(cat_var).dropna().nunique()
                                        if cat_unique_count_check != 2:
                                            raise ValueError(f"Safety check failed: Expected categorical variable {cat_var_name} to be binary, but found {cat_unique_count_check} unique values.")
                                        if cont_unique_count <= 50:  # Assuming continuous variables have more unique values
                                            raise ValueError(f"Safety check failed: Expected continuous variable {cont_var_name} to have many unique values, but found {cont_unique_count} unique values.")
                                        corr, p_value = weighted_point_biserial_correlation(cont_var, cat_var, w)
                                    else:
                                        placeholder.write(f"Using Pearson's correlation as an approximation for {var1} and {var2}: categorical variable {cat_var_name} has {cat_unique_count} unique values after subsetting (expected 2 for point-biserial).")
                                        correlation_type = "pearson_approx"
                                        corr, p_value = weighted_pearson_correlation(x, y, w)
                                else:
                                    if correlation_type == "pearson" or correlation_type == "pearson_approx":
                                        corr, p_value = weighted_pearson_correlation(x, y, w)
                                    elif correlation_type == "cramers_v":
                                        corr, p_value = weighted_cramers_v(x, y, w)
                                
                                if st.session_state.correlation_matrix_used_brr:
                                    placeholder.write("Calculating standard error using replicate weights...")
                                    # Pass work_done as a list to allow modification in the function
                                    work_done_ref = [work_done]
                                    # For point-biserial, pass cont_var and cat_var in the correct order
                                    if correlation_type == "point_biserial":
                                        se = compute_brr_se_correlation(cont_var, cat_var, w, replicate_weight_cols, pair_data, total_work, work_done_ref, work_per_brr, placeholder, progress_placeholder, correlation_type, cat_var_index)
                                    else:
                                        se = compute_brr_se_correlation(x, y, w, replicate_weight_cols, pair_data, total_work, work_done_ref, work_per_brr, placeholder, progress_placeholder, correlation_type, cat_var_index)
                                    work_done = work_done_ref[0]
                                    p_value = compute_brr_p_value(corr, se, len(pair_data))
                                else:
                                    se = np.nan  # Not used in simple method
                                    work_done += work_per_pv
                                    completed_pv_iterations += 1
                                    progress = min(work_done / total_work, 0.7)
                                    progress_placeholder.progress(progress)
                                
                                all_corrs.append(corr)
                                all_p_values.append(p_value)
                                all_se.append(se)
                                
                                if not st.session_state.correlation_matrix_used_brr:
                                    completed_pv_iterations += 1
                                    progress = min(work_done / total_work, 0.7)
                                    progress_placeholder.progress(progress)
                        
                            # Combine results
                            if num_pvs == 1:
                                placeholder.write(f"Combining results for {selected_labels[i]} and {selected_labels[j]} (single iteration, no plausible values)...")
                                combined_corr = all_corrs[0]
                                combined_p_value = all_p_values[0]
                                combined_se = all_se[0]
                            else:
                                placeholder.write(f"Combining results for {selected_labels[i]} and {selected_labels[j]}...")
                                combined_corr, combined_se, combined_p_value = apply_rubins_rules_correlations(all_corrs, all_se, len(pair_data))
                            
                            # Store in matrices
                            corr_matrix[i, j] = combined_corr
                            corr_matrix[j, i] = combined_corr
                            p_matrix[i, j] = combined_p_value
                            p_matrix[j, i] = combined_p_value
                    
                    # Update progress bar after correlation computation
                    progress_placeholder.progress(0.9)  # 90% after rendering the table
                    
                    # Store results in session state
                    st.session_state.correlation_matrix_results = {
                        'labels': selected_labels,
                        'codes': selected_codes,
                        'corr_matrix': corr_matrix,
                        'p_matrix': p_matrix,
                        'corr_types': corr_types,
                        'type_to_letter': type_to_letter
                    }
                    st.session_state.correlation_matrix_completed = True
                    
                    # Render the correlation matrix
                    placeholder.write("Rendering correlation matrix...")
                    matrix_height = len(selected_labels) * 90 + 30
                    table_html = render_correlation_matrix(selected_labels, corr_matrix, p_matrix, corr_types, type_to_letter)
                    components.html(table_html, height=matrix_height, scrolling=True)
                    placeholder.write("Correlation matrix analysis completed.")
                    
                    # Generate scatter plots for all pairs in a 2-column grid
                    st.header("Scatter plots for all variable pairs:")
                    variable_pairs = list(itertools.combinations(range(n_vars), 2))
                    placeholder.write(f"Generating scatter plots for {len(variable_pairs)} pairs of variables.")
                    
                    for idx in range(0, len(variable_pairs), 2):
                        # Create a row with 2 columns
                        cols = st.columns(2)
                        
                        # First plot in the row
                        i, j = variable_pairs[idx]
                        var1_label = selected_labels[i]
                        var2_label = selected_labels[j]
                        var1_code = selected_codes[i]
                        var2_code = selected_codes[j]
                        corr = corr_matrix[i, j]
                        corr_type = corr_types.get((i, j), "unknown")
                        corr_type_display = {
                            "pearson": "Pearson",
                            "point_biserial": "Point-Biserial",
                            "cramers_v": "Cramér's V",
                            "pearson_approx": "Pearson (approx)"
                        }.get(corr_type, "Unknown")
                        
                        # Handle plausible value domains
                        if var1_code in pv_domains:
                            var1_data = df[pv_domains[var1_code][0]]
                        else:
                            var1_data = df[var1_code]
                        
                        if var2_code in pv_domains:
                            var2_data = df[pv_domains[var2_code][0]]
                        else:
                            var2_data = df[var2_code]
                        
                        # Create plot dataframe step-by-step
                        data_dict = {
                            var1_code: var1_data,
                            var2_code: var2_data,
                            'weights': df['W_FSTUWT']
                        }
                        plot_df = pd.DataFrame(data_dict)
                        plot_df = plot_df.dropna()
                        
                        # Create scatter plot
                        fig1 = create_weighted_scatter(plot_df, var1_code, var2_code, 
                                                      plot_df['weights'], var1_label, var2_label, corr, corr_type_display)
                        with cols[0]:
                            st.pyplot(fig1)
                        
                        # Second plot in the row (if available)
                        if idx + 1 < len(variable_pairs):
                            i, j = variable_pairs[idx + 1]
                            var1_label = selected_labels[i]
                            var2_label = selected_labels[j]
                            var1_code = selected_codes[i]
                            var2_code = selected_codes[j]
                            corr = corr_matrix[i, j]
                            corr_type = corr_types.get((i, j), "unknown")
                            corr_type_display = {
                                "pearson": "Pearson",
                                "point_biserial": "Point-Biserial",
                                "cramers_v": "Cramér's V",
                                "pearson_approx": "Pearson (approx)"
                            }.get(corr_type, "Unknown")
                            
                            # Handle plausible value domains
                            if var1_code in pv_domains:
                                var1_data = df[pv_domains[var1_code][0]]
                            else:
                                var1_data = df[var1_code]
                            
                            if var2_code in pv_domains:
                                var2_data = df[pv_domains[var2_code][0]]
                            else:
                                var2_data = df[var2_code]
                            
                            # Create plot dataframe step-by-step
                            data_dict = {
                                var1_code: var1_data,
                                var2_code: var2_data,
                                'weights': df['W_FSTUWT']
                            }
                            plot_df = pd.DataFrame(data_dict)
                            plot_df = plot_df.dropna()
                            
                            # Create scatter plot
                            fig2 = create_weighted_scatter(plot_df, var1_code, var2_code, 
                                                          plot_df['weights'], var1_label, var2_label, corr, corr_type_display)
                            with cols[1]:
                                st.pyplot(fig2)
                    
                    # Update progress bar to 100% after scatter plots
                    progress_placeholder.progress(1.0)
                    placeholder.empty()  # Clear the placeholder after completion
            except Exception as e:
                placeholder.empty()
                st.error(f"Error computing correlation matrix: {str(e)}")
                st.session_state.correlation_matrix_completed = False
                st.session_state.correlation_matrix_results = None
        elif st.session_state.correlation_matrix_results and st.session_state.correlation_matrix_completed:
            if selected_var_labels:
                results = st.session_state.correlation_matrix_results
                matrix_height = len(results['labels']) * 90 + 30
                table_html = render_correlation_matrix(
                    results['labels'],
                    results['corr_matrix'],
                    results['p_matrix'],
                    results['corr_types'],
                    results['type_to_letter']
                )
                components.html(table_html, height=matrix_height, scrolling=True)
                st.write("Correlation matrix analysis completed.")
            else:
                st.write("Please select at least two variables to compute the correlation matrix.")
        else:
            st.write("Please select at least two variables and click 'Run Analysis' to compute the correlation matrix.")

# Instructions section
st.header("Instructions")
st.markdown("""
- **Select Variables**: Choose two or more variables (domains or numeric variables) from the dropdown menus. Plausible value domains (e.g., Mathematics score, Reading score) will use all 10 plausible values for analysis.
- **Run Analysis**: Click "Run Analysis" to perform the weighted correlation matrix analysis. Analyses involving plausible values will be combined using Rubin's rules.
- **View Results**: Results are displayed in an APA-style table with correlations rounded to 2 decimal places. Superscript letters indicate the correlation type for each pair, explained in the note below the table.
- **Navigate**: Use the sidebar to switch between different analysis types or return to the main page to upload a new dataset.
""")