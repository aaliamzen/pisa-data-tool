import streamlit as st
import pandas as pd
import numpy as np
import re
import streamlit.components.v1 as components
from scipy.stats import t

# Streamlit app configuration
st.set_page_config(page_title="Correlation Matrix - PISA Data Exploration Tool", layout="wide")

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

# Function to generate sample write-up for correlation matrix
def generate_correlation_matrix_write_up(selected_labels, selected_codes, corr_matrix, p_matrix, used_brr, num_pvs):
    write_up = (
        f"A weighted correlation matrix analysis {'using BRR standard errors' if used_brr else 'using a simple method'} "
        f"was conducted on {len(selected_labels)} variables, using {num_pvs} plausible values combined via Rubin's rules "
        f"where applicable. The variables analyzed were: {', '.join([f'{label} ({code})' for label, code in zip(selected_labels, selected_codes)])}.\n\n"
    )
    
    significant_pairs = []
    for i in range(len(selected_labels)):
        for j in range(i + 1, len(selected_labels)):
            corr = corr_matrix[i, j]
            p_value = p_matrix[i, j]
            if np.isnan(p_value) or p_value >= 0.05:
                continue
            corr_display = f"{corr:.3f}"
            p_display = "< .001" if p_value < 0.001 else f"= {p_value:.3f}"
            significant_pairs.append(
                f"{selected_labels[i]} ({selected_codes[i]}) and {selected_labels[j]} ({selected_codes[j]}) "
                f"showed a significant correlation of {corr_display} (p {p_display})"
            )
    
    if significant_pairs:
        write_up += "Significant correlations (p < 0.05) were found for the following pairs:\n- " + "\n- ".join(significant_pairs) + "."
    else:
        write_up += "No significant correlations (p < 0.05) were found among the selected variables."
    
    return write_up

# Function to render correlation matrix as HTML table
def render_correlation_matrix(selected_labels, corr_matrix, p_matrix):
    html_content = """
    <style>
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
                corr_display = f"{corr:.3f}" if not np.isnan(corr) else "-"
                sig_display = "**" if p_value < 0.01 else "*" if p_value < 0.05 else "" if not np.isnan(p_value) else ""
                cell_content = f"{corr_display}{sig_display}"
            row += f"<td>{cell_content}</td>"
        row += "</tr>"
        data_rows += row
    
    full_html = html_content.replace("{{header_row}}", header_row).replace("{{data_rows}}", data_rows)
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
if 'correlation_matrix_show_write_up' not in st.session_state:
    st.session_state.correlation_matrix_show_write_up = False
if 'correlation_matrix_write_up_content' not in st.session_state:
    st.session_state.correlation_matrix_write_up_content = None
if 'correlation_matrix_selected_vars' not in st.session_state:
    st.session_state.correlation_matrix_selected_vars = None
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
        # Select Variables (default to MATH domain and ESCS)
        st.write("Select variables for correlation matrix (default: Mathematics score, ESCS):")
        all_var_options = domain_options + unique_var_labels
        default_vars = []
        if "Mathematics score" in all_var_options:
            default_vars.append("Mathematics score")
        escs_label = variable_labels.get("ESCS", "ESCS")
        if escs_label in seen_labels and escs_label in all_var_options:
            default_vars.append(escs_label)
        elif all_var_options:
            default_vars.append(all_var_options[0])
        
        selected_var_labels = st.multiselect(
            "Variables",
            all_var_options,
            default=default_vars,
            key="matrix_vars"
        )
        
        # Split selected variables into domains and regular variables
        selected_domains = []
        selected_vars = []
        for label in selected_var_labels:
            if label in label_to_domain:
                domain = label_to_domain[label]
                selected_domains.append(domain)
            else:
                selected_vars.append(label_to_var[label])
        
        # Combine selected variables and domains
        all_selected_labels = [label for label in selected_var_labels if label not in label_to_domain]
        for domain in selected_domains:
            all_selected_labels.append(domain_to_label.get(domain, domain))
        
        run_analysis = st.button("Run Analysis", key="run_correlation_matrix")
        
        # Update session state with selections
        if selected_var_labels != st.session_state.correlation_matrix_selected_vars:
            st.session_state.correlation_matrix_selected_vars = selected_var_labels
            if not run_analysis:  # Only reset if not running analysis
                st.session_state.correlation_matrix_results = None
                st.session_state.correlation_matrix_completed = False
                st.session_state.correlation_matrix_show_write_up = False
                st.session_state.correlation_matrix_write_up_content = None
                st.session_state.correlation_matrix_used_brr = False
        
        if run_analysis and (selected_vars or selected_domains) and len(selected_var_labels) >= 2:
            try:
                if 'W_FSTUWT' not in df.columns:
                    st.error("Final student weight (W_FSTUWT) not found in the dataset.")
                else:
                    # Check for replicate weights availability
                    replicate_weight_cols = [f"W_FSTURWT{i}" for i in range(1, 81)]
                    missing_weights = [col for col in replicate_weight_cols if col not in df.columns]
                    st.session_state.correlation_matrix_used_brr = len(missing_weights) == 0  # Use BRR if no replicate weights are missing
                    
                    # Prepare columns for data filtering
                    columns = ['W_FSTUWT']
                    if st.session_state.correlation_matrix_used_brr:
                        columns += replicate_weight_cols
                    for domain in selected_domains:
                        columns.extend(pv_domains[domain])
                    columns.extend(selected_vars)
                    
                    corr_data = df[columns].dropna()
                    if len(corr_data) < 2:
                        st.warning("Insufficient non-missing data for correlation matrix analysis.")
                        st.session_state.correlation_matrix_completed = False
                        st.session_state.correlation_matrix_show_write_up = False
                        st.session_state.correlation_matrix_write_up_content = None
                        st.session_state.correlation_matrix_used_brr = False
                    else:
                        st.write(f"Dataset size after dropping NA: {len(corr_data)} rows")
                        # Determine the number of variables (including domains)
                        selected_codes = []
                        all_vars = []
                        for label in selected_var_labels:
                            if label in label_to_domain:
                                domain = label_to_domain[label]
                                selected_codes.append(domain)
                                all_vars.append(pv_domains[domain])
                            else:
                                selected_codes.append(label_to_var[label])
                                all_vars.append([label_to_var[label]])
                        
                        # Compute the maximum number of plausible values
                        max_pvs = max([len(vars) for vars in all_vars])
                        
                        # Initialize correlation and p-value matrices
                        n_vars = len(selected_var_labels)
                        corr_matrix = np.ones((n_vars, n_vars))
                        p_matrix = np.zeros((n_vars, n_vars))
                        
                        # Progress bar for PV iterations
                        pv_progress = st.progress(0)
                        total_iterations = (n_vars * (n_vars - 1)) // 2 * max_pvs
                        iteration_count = 0
                        
                        # Compute correlations for each pair of variables
                        for i in range(n_vars):
                            for j in range(i + 1, n_vars):
                                var1_list = all_vars[i]
                                var2_list = all_vars[j]
                                var1_is_pv = len(var1_list) > 1
                                var2_is_pv = len(var2_list) > 1
                                num_pvs = max(len(var1_list), len(var2_list))
                                
                                all_corrs = []
                                all_se = []
                                
                                for pv_idx in range(num_pvs):
                                    var1 = var1_list[pv_idx % len(var1_list)]
                                    var2 = var2_list[pv_idx % len(var2_list)]
                                    st.write(f"Processing correlation between {var1} and {var2}...")
                                    
                                    x = corr_data[var1].values
                                    y = corr_data[var2].values
                                    w = corr_data['W_FSTUWT'].values
                                    corr, p_value = weighted_correlation(x, y, w)
                                    if st.session_state.correlation_matrix_used_brr:
                                        st.write("Calculating standard error using replicate weights...")
                                        brr_progress = st.progress(0)
                                        se = compute_brr_se_correlation(x, y, replicate_weight_cols, corr_data, brr_progress)
                                        p_value = compute_brr_p_value(corr, se, len(corr_data))
                                    else:
                                        se = np.nan  # Not used in simple method
                                    
                                    all_corrs.append(corr)
                                    all_se.append(se)
                                    
                                    iteration_count += 1
                                    pv_progress.progress(iteration_count / total_iterations)
                        
                                # Combine results using Rubin's rules
                                st.write(f"Combining results for {all_selected_labels[i]} and {all_selected_labels[j]}...")
                                combined_corr, combined_se, combined_p_value = apply_rubins_rules_correlations(all_corrs, all_se, len(corr_data))
                                
                                # Store in matrices
                                corr_matrix[i, j] = combined_corr
                                corr_matrix[j, i] = combined_corr
                                p_matrix[i, j] = combined_p_value
                                p_matrix[j, i] = combined_p_value
                        
                        # Store results in session state
                        st.session_state.correlation_matrix_results = {
                            'labels': all_selected_labels,
                            'codes': selected_codes,
                            'corr_matrix': corr_matrix,
                            'p_matrix': p_matrix
                        }
                        st.session_state.correlation_matrix_completed = True
                        st.session_state.correlation_matrix_show_write_up = False
                        st.session_state.correlation_matrix_write_up_content = None
                        
                        # Render the correlation matrix
                        st.write("Rendering correlation matrix...")
                        table_html = render_correlation_matrix(all_selected_labels, corr_matrix, p_matrix)
                        components.html(table_html, height=400, scrolling=True)
                        st.write("Correlation matrix analysis completed.")
            except Exception as e:
                st.error(f"Error computing correlation matrix: {str(e)}")
                st.session_state.correlation_matrix_completed = False
                st.session_state.correlation_matrix_show_write_up = False
                st.session_state.correlation_matrix_write_up_content = None
                st.session_state.correlation_matrix_used_brr = False
        elif st.session_state.correlation_matrix_results and st.session_state.correlation_matrix_completed:
            if st.session_state.correlation_matrix_selected_vars == selected_var_labels:
                results = st.session_state.correlation_matrix_results
                if results is not None:
                    table_html = render_correlation_matrix(
                        results['labels'],
                        results['corr_matrix'],
                        results['p_matrix']
                    )
                    components.html(table_html, height=400, scrolling=True)
                    st.write("Correlation matrix analysis completed.")
                else:
                    st.write("Correlation matrix results are not available. Please run the analysis again.")
            else:
                st.write("Variable selection has changed. Please click 'Run Analysis' to compute the correlation matrix with the new variables.")
        else:
            st.write("Please click 'Run Analysis' to compute the correlation matrix.")
        
        if st.session_state.correlation_matrix_results and st.session_state.correlation_matrix_completed:
            if (st.session_state.correlation_matrix_selected_vars == selected_var_labels and
                st.session_state.correlation_matrix_results is not None):
                if st.button("Generate Write-Up", key="writeup_correlation_matrix"):
                    results = st.session_state.correlation_matrix_results
                    num_pvs = max([len(pv_domains[label_to_domain[label]]) if label in label_to_domain else 1 for label in selected_var_labels])
                    write_up = generate_correlation_matrix_write_up(
                        results['labels'],
                        results['codes'],
                        results['corr_matrix'],
                        results['p_matrix'],
                        st.session_state.correlation_matrix_used_brr,
                        num_pvs
                    )
                    st.session_state.correlation_matrix_show_write_up = True
                    st.session_state.correlation_matrix_write_up_content = write_up
            else:
                st.write("Correlation matrix results are not available or variables have changed. Please run the analysis again to generate the write-up.")
        
        # Display the write-up if it exists
        if st.session_state.correlation_matrix_show_write_up and st.session_state.correlation_matrix_write_up_content:
            st.markdown("### Sample Write-Up", unsafe_allow_html=True)
            st.markdown(st.session_state.correlation_matrix_write_up_content, unsafe_allow_html=True)

# Instructions section
st.header("Instructions")
st.markdown("""
- **Select Variables**: Choose two or more variables (domains or numeric variables) from the dropdown menus. Plausible value domains (e.g., Mathematics score, Reading score) will use all 10 plausible values for analysis.
- **Run Analysis**: Click "Run Analysis" to perform the weighted correlation matrix analysis. Analyses involving plausible values will be combined using Rubin's rules.
- **Generate Write-Up**: After running the analysis, click "Generate Write-Up" to see a sample description of the results.
- **Navigate**: Use the sidebar to switch between different analysis types or return to the main page to upload a new dataset.
""")