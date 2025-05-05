import streamlit as st
import pandas as pd
import numpy as np
import re
import streamlit.components.v1 as components
from scipy.stats import t

# Streamlit app configuration
st.set_page_config(page_title="Bivariate Correlation - PISA Data Exploration Tool", layout="wide")

# Function to compute weighted correlation coefficient
def weighted_correlation(x, y, w):
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
        st.error(f"Error in weighted_correlation: {str(e)}")
        return np.nan, np.nan

# Function to compute BRR standard errors for correlation
def compute_brr_se_correlation(x, y, replicate_weights, corr_data, progress_bar):
    try:
        # Initial correlation with final student weights for reference
        main_corr, _ = weighted_correlation(x, y, corr_data['W_FSTUWT'].values)
        if np.isnan(main_corr):
            return np.nan
        
        # Compute correlations for each replicate weight
        replicate_corrs = []
        total_weights = len(replicate_weights)
        for idx, weight_col in enumerate(replicate_weights):
            # Create a DataFrame with only the weights to handle NA dropping
            data = corr_data[[weight_col, 'W_FSTUWT']].copy()
            data = data.dropna()
            if len(data) < 2:
                continue
            
            # Subset x and y to match the non-NA indices of the weights
            indices = data.index
            pos = corr_data.index.get_indexer(indices)
            pos = pos[pos != -1]  # Remove invalid indices
            if len(pos) < 2:
                continue
            x_rep = x[pos]
            y_rep = y[pos]
            w_rep = data[weight_col].values
            
            # Compute correlation with replicate weights
            rep_corr, _ = weighted_correlation(x_rep, y_rep, w_rep)
            if not np.isnan(rep_corr):
                replicate_corrs.append(rep_corr)
            # Update progress bar
            progress_bar.progress((idx + 1) / total_weights)
        
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
        combined_corr = np.mean(corrs_array)
        
        # Within-imputation variance
        within_var = np.mean(se_array**2)
        
        # Between-imputation variance
        between_var = np.var(corrs_array, ddof=1)
        
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

# Function to generate sample write-up for correlation
def generate_correlation_write_up(var1_label, var1_code, var2_label, var2_code, corr, p_value, used_brr, num_pvs):
    corr_display = f"{corr:.3f}" if not np.isnan(corr) else "N/A"
    p_display = "< .001" if p_value < 0.001 else f"= {p_value:.3f}" if not np.isnan(p_value) else "N/A"
    method_note = "using BRR standard errors" if used_brr else "using a simple method"
    
    # Conditionally include PV mention based on num_pvs
    if num_pvs == 1:
        write_up = (
            f"A weighted correlation analysis {method_note} was conducted between {var1_label} ({var1_code}) and "
            f"{var2_label} ({var2_code}). The correlation coefficient was {corr_display}, with a p-value {p_display}."
        )
    else:
        write_up = (
            f"A weighted correlation analysis {method_note} was conducted between {var1_label} ({var1_code}) and "
            f"{var2_label} ({var2_code}), using {num_pvs} plausible values combined via Rubin's rules. "
            f"The correlation coefficient was {corr_display}, with a p-value {p_display}."
        )
    return write_up

# Function to render correlation results as HTML table
def render_correlation_table(var1_label, var1_code, var2_label, var2_code, corr, p_value):
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
    }
    .corr-table th:first-child, .corr-table td:first-child {
        width: 200px !important;
        text-align: left;
        white-space: normal;
        overflow-wrap: break-word;
    }
    .corr-table th:not(:first-child), .corr-table td:not(:first-child) {
        width: 100px !important;
        text-align: center;
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
    </style>
    <div class="corr-table-container">
        <div class="corr-table-title">Table 1</div>
        <div class="corr-table-subtitle">Correlation Between {{var1_label}} and {{var2_label}}</div>
        <table class="corr-table">
            <tr class="corr-table-header">
                <th>Variable Pair</th>
                <th>Correlation</th>
                <th>p-value</th>
                <th>Significance</th>
            </tr>
            <tr class="corr-table-last-row">
                <th>{{var1_label}} and {{var2_label}}</th>
                <td>{{corr_display}}</td>
                <td>{{p_display}}</td>
                <td>{{sig_display}}</td>
            </tr>
        </table>
    </div>
    """
    corr_display = f"{corr:.3f}" if not np.isnan(corr) else "-"
    p_display = "< .001" if p_value < 0.001 else f"{p_value:.3f}" if not np.isnan(p_value) else "-"
    sig_display = "**" if p_value < 0.01 else "*" if p_value < 0.05 else "" if not np.isnan(p_value) else "-"
    
    full_html = html_content.replace("{{var1_label}}", var1_label).replace("{{var2_label}}", var2_label).replace("{{corr_display}}", corr_display).replace("{{p_display}}", p_display).replace("{{sig_display}}", sig_display)
    return full_html

# Access data from session state
df = st.session_state.get('df', None)
variable_labels = st.session_state.get('variable_labels', {})
value_labels = st.session_state.get('value_labels', {})
visible_columns = st.session_state.get('visible_columns', [])

# Initialize session state for this page
if 'correlation_results' not in st.session_state:
    st.session_state.correlation_results = None
if 'correlation_completed' not in st.session_state:
    st.session_state.correlation_completed = False
if 'correlation_show_write_up' not in st.session_state:
    st.session_state.correlation_show_write_up = False
if 'correlation_write_up_content' not in st.session_state:
    st.session_state.correlation_write_up_content = None
if 'correlation_var1' not in st.session_state:
    st.session_state.correlation_var1 = None
if 'correlation_var2' not in st.session_state:
    st.session_state.correlation_var2 = None
if 'correlation_used_brr' not in st.session_state:
    st.session_state.correlation_used_brr = False

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
        st.warning("At least two numeric variables or domains are required for correlation analysis.")
    else:
        # Select Variable 1 (default to MATH domain)
        st.write("Select first variable (domain or numeric variable):")
        all_var_options = domain_options + unique_var_labels
        default_var1_label = "Mathematics score"
        if default_var1_label not in all_var_options:
            st.warning(f"Default first variable '{default_var1_label}' not found in available options: {all_var_options}. Falling back to first variable.")
            default_index1 = 0
        else:
            default_index1 = all_var_options.index(default_var1_label)
        
        var1_label = st.selectbox(
            "Variable 1",
            all_var_options,
            index=default_index1,
            key="var1"
        )
        
        # Determine if Variable 1 is a domain or a regular variable
        if var1_label in label_to_domain:
            var1_domain = label_to_domain[var1_label]
            var1_vars = pv_domains[var1_domain]
            var1_is_pv = True
        else:
            var1_vars = [label_to_var[var1_label]]
            var1_is_pv = False
        
        # Select Variable 2 (default to ESCS)
        st.write("Select second variable (default: ESCS):")
        var2_options = [label for label in all_var_options if label != var1_label]
        default_var2_label = variable_labels.get("ESCS", "ESCS")
        if default_var2_label in seen_labels and default_var2_label in var2_options:
            default_index2 = var2_options.index(default_var2_label)
        else:
            default_index2 = 0 if var2_options else None
        
        if not var2_options:
            st.warning("No second variable available to select. Please ensure there are at least two variables in the dataset.")
        else:
            var2_label = st.selectbox(
                "Variable 2",
                var2_options,
                index=default_index2,
                key="var2"
            )
            
            # Determine if Variable 2 is a domain or a regular variable
            if var2_label in label_to_domain:
                var2_domain = label_to_domain[var2_label]
                var2_vars = pv_domains[var2_domain]
                var2_is_pv = True
            else:
                var2_vars = [label_to_var[var2_label]]
                var2_is_pv = False
            
            run_analysis = st.button("Run Analysis", key="run_correlation")
            
            # Update session state with selections
            if (var1_label != st.session_state.correlation_var1 or 
                var2_label != st.session_state.correlation_var2):
                st.session_state.correlation_var1 = var1_label
                st.session_state.correlation_var2 = var2_label
                if not run_analysis:  # Only reset if not running analysis
                    st.session_state.correlation_results = None
                    st.session_state.correlation_completed = False
                    st.session_state.correlation_show_write_up = False
                    st.session_state.correlation_write_up_content = None
                    st.session_state.correlation_used_brr = False
            
            if run_analysis and var1_vars and var2_vars:
                try:
                    if 'W_FSTUWT' not in df.columns:
                        st.error("Final student weight (W_FSTUWT) not found in the dataset.")
                    else:
                        # Check for replicate weights availability
                        replicate_weight_cols = [f"W_FSTURWT{i}" for i in range(1, 81)]
                        missing_weights = [col for col in replicate_weight_cols if col not in df.columns]
                        st.session_state.correlation_used_brr = len(missing_weights) == 0  # Use BRR if no replicate weights are missing
                        
                        # Prepare columns for data filtering
                        columns = ['W_FSTUWT']
                        if st.session_state.correlation_used_brr:
                            columns += replicate_weight_cols
                        if var1_is_pv:
                            columns.extend(var1_vars)
                        else:
                            columns.append(var1_vars[0])
                        if var2_is_pv:
                            columns.extend(var2_vars)
                        else:
                            columns.append(var2_vars[0])
                        
                        corr_data = df[columns].dropna()
                        if len(corr_data) < 2:
                            st.warning("Insufficient non-missing data for correlation analysis (at least 2 observations required after dropping missing values).")
                            st.session_state.correlation_completed = False
                            st.session_state.correlation_show_write_up = False
                            st.session_state.correlation_write_up_content = None
                            st.session_state.correlation_used_brr = False
                        else:
                            st.write(f"Dataset size after dropping NA: {len(corr_data)} rows")
                            num_pvs = max(len(var1_vars) if var1_is_pv else 1, len(var2_vars) if var2_is_pv else 1)
                            
                            # Prepare lists to store results across PV combinations
                            all_corrs = []
                            all_p_values = []  # Store p-values directly
                            all_se = []
                            
                            # Progress bar for PV iterations
                            pv_progress = st.progress(0)
                            total_pv_iterations = num_pvs * num_pvs if var1_is_pv and var2_is_pv else num_pvs
                            iteration_count = 0
                            
                            # Iterate over all PV combinations
                            for pv1_idx in range(len(var1_vars) if var1_is_pv else 1):
                                pv1_num = pv1_idx + 1
                                x_var = var1_vars[pv1_idx] if var1_is_pv else var1_vars[0]
                                for pv2_idx in range(len(var2_vars) if var2_is_pv else 1):
                                    pv2_num = pv2_idx + 1
                                    y_var = var2_vars[pv2_idx] if var2_is_pv else var2_vars[0]
                                    st.write(f"Processing correlation between {x_var} and {y_var}...")
                                    
                                    # Compute correlation
                                    x = corr_data[x_var].values
                                    y = corr_data[y_var].values
                                    w = corr_data['W_FSTUWT'].values
                                    corr, p_value = weighted_correlation(x, y, w)
                                    if st.session_state.correlation_used_brr:
                                        st.write("Calculating standard error using replicate weights...")
                                        brr_progress = st.progress(0)
                                        se = compute_brr_se_correlation(x, y, replicate_weight_cols, corr_data, brr_progress)
                                        p_value = compute_brr_p_value(corr, se, len(corr_data))
                                    else:
                                        se = np.nan  # Not used in simple method
                                    
                                    # Store results
                                    all_corrs.append(corr)
                                    all_p_values.append(p_value)
                                    all_se.append(se)
                                    
                                    # Update progress
                                    iteration_count += 1
                                    pv_progress.progress(iteration_count / total_pv_iterations)
                            
                            # Combine results
                            if num_pvs == 1:
                                # For non-PV variables, use the single correlation and p-value directly
                                st.write("Combining results (single iteration, no plausible values)...")
                                combined_corr = all_corrs[0]
                                combined_p_value = all_p_values[0]
                                combined_se = all_se[0]
                            else:
                                # For PV variables, apply Rubin's rules
                                st.write("Combining results across plausible values using Rubin's rules...")
                                combined_corr, combined_se, combined_p_value = apply_rubins_rules_correlations(all_corrs, all_se, len(corr_data))
                            
                            # Store results in session state
                            st.session_state.correlation_results = {
                                'var1_label': var1_label,
                                'var1_code': var1_domain if var1_is_pv else var1_vars[0],
                                'var2_label': var2_label,
                                'var2_code': var2_domain if var2_is_pv else var2_vars[0],
                                'corr': combined_corr,
                                'p_value': combined_p_value
                            }
                            st.session_state.correlation_completed = True
                            st.session_state.correlation_show_write_up = False
                            st.session_state.correlation_write_up_content = None
                            
                            # Render the correlation table
                            st.write("Rendering correlation table...")
                            table_html = render_correlation_table(
                                var1_label,
                                var1_domain if var1_is_pv else var1_vars[0],
                                var2_label,
                                var2_domain if var2_is_pv else var2_vars[0],
                                combined_corr,
                                combined_p_value
                            )
                            components.html(table_html, height=200, scrolling=True)
                            st.write("Correlation analysis completed.")
                except Exception as e:
                    st.error(f"Error computing correlation: {str(e)}")
                    st.session_state.correlation_completed = False
                    st.session_state.correlation_show_write_up = False
                    st.session_state.correlation_write_up_content = None
                    st.session_state.correlation_used_brr = False
            elif st.session_state.correlation_results and st.session_state.correlation_completed:
                if (st.session_state.correlation_var1 == var1_label and 
                    st.session_state.correlation_var2 == var2_label):
                    results = st.session_state.correlation_results
                    if results is not None:
                        table_html = render_correlation_table(
                            results['var1_label'],
                            results['var1_code'],
                            results['var2_label'],
                            results['var2_code'],
                            results['corr'],
                            results['p_value']
                        )
                        components.html(table_html, height=200, scrolling=True)
                        st.write("Correlation analysis completed.")
                    else:
                        st.write("Correlation results are not available. Please run the analysis again.")
                else:
                    st.write("Variable selection has changed. Please click 'Run Analysis' to compute the correlation with the new variables.")
            else:
                st.write("Please click 'Run Analysis' to compute the correlation.")
            
            if st.session_state.correlation_results and st.session_state.correlation_completed:
                if (st.session_state.correlation_var1 == var1_label and 
                    st.session_state.correlation_var2 == var2_label and
                    st.session_state.correlation_results is not None):
                    if st.button("Generate Write-Up", key="writeup_correlation"):
                        results = st.session_state.correlation_results
                        num_pvs = max(len(var1_vars) if var1_is_pv else 1, len(var2_vars) if var2_is_pv else 1)
                        write_up = generate_correlation_write_up(
                            results['var1_label'],
                            results['var1_code'],
                            results['var2_label'],
                            results['var2_code'],
                            results['corr'],
                            results['p_value'],
                            st.session_state.correlation_used_brr,
                            num_pvs
                        )
                        st.session_state.correlation_show_write_up = True
                        st.session_state.correlation_write_up_content = write_up
                else:
                    st.write("Correlation results are not available or variables have changed. Please run the analysis again to generate the write-up.")
            
            # Display the write-up if it exists
            if st.session_state.correlation_show_write_up and st.session_state.correlation_write_up_content:
                st.markdown("### Sample Write-Up", unsafe_allow_html=True)
                st.markdown(st.session_state.correlation_write_up_content, unsafe_allow_html=True)

# Instructions section
st.header("Instructions")
st.markdown("""
- **Select Variables**: Choose two variables (domains or numeric variables) from the dropdown menus. Plausible value domains (e.g., Mathematics score, Reading score) will use all 10 plausible values for analysis.
- **Run Analysis**: Click "Run Analysis" to perform the weighted correlation analysis. Analyses involving plausible values will be combined using Rubin's rules.
- **Generate Write-Up**: After running the analysis, click "Generate Write-Up" to see a sample description of the results.
- **Navigate**: Use the sidebar to switch between different analysis types or return to the main page to upload a new dataset.
""")