import streamlit as st
import pandas as pd
import numpy as np
import re
import statsmodels.api as sm
import streamlit.components.v1 as components
from scipy.stats import t

# Streamlit app configuration
st.set_page_config(page_title="Linear Regression - PISA Data Exploration Tool", layout="wide")

# Function to compute weighted linear regression coefficients
def weighted_linear_regression(X, y, w):
    try:
        X = sm.add_constant(X)  # Add intercept
        model = sm.WLS(y, X, weights=w)
        results = model.fit()
        return results
    except Exception as e:
        st.error(f"Error in weighted_linear_regression: {str(e)}")
        return None

# Function to compute weighted standardization
def weighted_standardize(data, weights):
    try:
        if not all(weights > 0):
            raise ValueError("Weights must be positive.")
        weighted_mean = np.sum(data * weights) / np.sum(weights)
        weighted_var = np.sum(weights * (data - weighted_mean)**2) / np.sum(weights)
        weighted_std = np.sqrt(weighted_var)
        if weighted_std == 0:
            return np.zeros_like(data)  # Avoid division by zero
        return (data - weighted_mean) / weighted_std
    except Exception as e:
        st.error(f"Error in weighted_standardize: {str(e)}")
        return np.zeros_like(data)

# Function to compute BRR standard errors for regression coefficients
def compute_brr_se_regression(X, y, replicate_weights, reg_data, progress_bar):
    try:
        # Initial regression with final student weights for reference
        X_full = sm.add_constant(X)  # Add intercept
        main_model = sm.WLS(y, X_full, weights=reg_data['W_FSTUWT'])
        main_results = main_model.fit()
        main_coefs = main_results.params
        
        # Compute coefficients for each replicate weight
        replicate_coefs = []
        total_weights = len(replicate_weights)
        for idx, weight_col in enumerate(replicate_weights):
            # Create a DataFrame with only the weights to handle NA dropping
            data = reg_data[[weight_col, 'W_FSTUWT']].copy()
            data = data.dropna()
            if len(data) < X.shape[1] + 1:  # Need at least as many observations as parameters
                continue
            
            # Subset X and y to match the non-NA indices of the weights
            indices = data.index
            # Convert indices to positions in X and y
            pos = reg_data.index.get_indexer(indices)
            pos = pos[pos != -1]  # Remove invalid indices
            if len(pos) < X.shape[1] + 1:  # Ensure enough observations remain
                continue
            X_rep = X[pos]
            y_rep = y[pos]
            w_rep = data[weight_col].values
            
            # Run regression with replicate weights
            X_rep = sm.add_constant(X_rep)  # Add intercept
            model = sm.WLS(y_rep, X_rep, weights=w_rep)
            results = model.fit()
            replicate_coefs.append(results.params)
            # Update progress bar
            progress_bar.progress((idx + 1) / total_weights)
        
        if not replicate_coefs:
            return np.array([np.nan] * len(main_coefs))
        
        # Convert to array and compute standard errors
        replicate_coefs = np.array(replicate_coefs)
        se = np.sqrt((1 / 80) * np.sum((replicate_coefs - main_coefs) ** 2, axis=0))
        return se
    except Exception as e:
        st.error(f"Error in compute_brr_se_regression: {str(e)}")
        return np.array([np.nan] * len(main_coefs))

# Function to compute p-values using BRR standard errors
def compute_brr_p_values(coefs, se, n, k):
    try:
        p_values = []
        df = n - k  # Degrees of freedom: n - number of parameters
        for c, s in zip(coefs, se):
            if np.isnan(s) or s == 0:
                p_values.append(np.nan)
            else:
                t_stat = c / s
                p_value = 2 * (1 - t.cdf(np.abs(t_stat), df=df))
                p_values.append(p_value)
        return np.array(p_values)
    except Exception as e:
        st.error(f"Error in compute_brr_p_values: {str(e)}")
        return np.array([np.nan] * len(coefs))

# Function to apply Rubin's rules for combining plausible value results
def apply_rubins_rules(coefs_list, se_list, n, k):
    try:
        # Convert lists to arrays for computation
        coefs_array = np.array(coefs_list)  # Shape: (num_pvs, num_params)
        se_array = np.array(se_list)        # Shape: (num_pvs, num_params)
        num_pvs = len(coefs_list)
        
        # Step 1: Compute the combined point estimate (average of coefficients)
        combined_coefs = np.mean(coefs_array, axis=0)
        
        # Step 2: Compute within-imputation variance (average of squared SEs)
        within_var = np.mean(se_array**2, axis=0)
        
        # Step 3: Compute between-imputation variance
        between_var = np.var(coefs_array, axis=0, ddof=1)
        
        # Step 4: Compute total variance
        total_var = within_var + (1 + 1/num_pvs) * between_var
        
        # Step 5: Compute combined standard error
        combined_se = np.sqrt(total_var)
        
        # Step 6: Compute p-values
        p_values = compute_brr_p_values(combined_coefs, combined_se, n, k)
        
        return combined_coefs, combined_se, p_values
    except Exception as e:
        st.error(f"Error in apply_rubins_rules: {str(e)}")
        return np.array([np.nan]), np.array([np.nan]), np.array([np.nan])

# Function to generate sample write-up for linear regression
def generate_regression_write_up(dep_label, dep_code, indep_labels, indep_codes, coefs, p_values, r_squared, adj_r_squared, aic, bic, f_stat, f_pvalue, df, used_brr, num_pvs):
    significant_effects = []
    for i, (label, code, coef, p) in enumerate(zip(indep_labels[1:], indep_codes[1:], coefs[1:], p_values[1:])):  # Skip intercept
        if np.isnan(p) or p >= 0.05:
            continue
        direction = "increase" if coef > 0 else "decrease"
        effect_size = abs(coef)
        p_display = "< .001" if p < 0.001 else f"= {p:.3f}"
        # List other predictors being controlled for
        other_predictors = [lbl for j, lbl in enumerate(indep_labels[1:]) if j != i]
        control_text = ", ".join(other_predictors) if other_predictors else "no other variables"
        significant_effects.append(
            f"A one-point increase in {label} ({code}) predicts a {effect_size:.3f} point {direction} in {dep_label} ({dep_code}) "
            f"when controlling for {control_text} (<i>p</i> {p_display})"
        )
    
    r2_display = f"{r_squared:.3f}"
    adj_r2_display = f"{adj_r_squared:.3f}"
    aic_display = f"{aic:.2f}"
    bic_display = f"{bic:.2f}"
    f_p_display = "< .001" if f_pvalue < 0.001 else f"= {f_pvalue:.3f}"
    method_note = "using BRR standard errors" if used_brr else "using a simple method"
    
    write_up = (
        f"A weighted linear regression analysis {method_note} (df = {df}) was conducted with {dep_label} ({dep_code}) "
        f"as the dependent variable and {len(indep_labels) - 1} independent variables, using {num_pvs} plausible values combined via Rubin's rules. "
    )
    
    if significant_effects:
        write_up += (
            f"The model identified {len(significant_effects)} significant predictor(s): "
            f"{'; '.join(significant_effects)}. "
        )
    else:
        write_up += "No statistically significant predictors were found. "
    
    write_up += (
        f"The overall model was {'statistically significant' if f_pvalue < 0.05 else 'not statistically significant'} "
        f"(<i>F</i> = {f_stat:.2f}, <i>p</i> {f_p_display}), with fit measures of "
        f"<i>R</i>² = {r2_display}, Adjusted <i>R</i>² = {adj_r2_display}, "
        f"AIC = {aic_display}, and BIC = {bic_display}."
    )
    
    return write_up

# Function to render regression results as HTML table
def render_regression_table(dep_label, dep_code, indep_labels, indep_codes, coefs, std_coefs, p_values, r_squared, adj_r_squared, df):
    html_content = """
    <style>
    .reg-table-container {
        display: inline-block;
        overflow-x: auto;
        scrollbar-width: thin;
        min-width: 0;
        margin: 20px 0;
    }
    .reg-table-container::-webkit-scrollbar {
        height: 8px;
    }
    .reg-table-container::-webkit-scrollbar-thumb {
        background-color: #888;
        border-radius: 4px;
    }
    .reg-table {
        table-layout: fixed;
        border-collapse: collapse;
        font-size: 14px;
        margin: 0;
    }
    .reg-table th, .reg-table td {
        border: none;
        padding: 8px;
        box-sizing: border-box;
    }
    .reg-table th:first-child, .reg-table td:first-child {
        width: 200px !important;
        text-align: left;
        white-space: normal;
        overflow-wrap: break-word;
    }
    .reg-table th:not(:first-child), .reg-table td:not(:first-child) {
        width: 100px !important;
        text-align: center;
    }
    .reg-table tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    .reg-table-title {
        font-size: 16px;
        font-weight: bold;
        text-align: left;
        margin-bottom: 5px;
    }
    .reg-table-subtitle {
        font-size: 16px;
        font-style: italic;
        text-align: left;
        margin-bottom: 10px;
    }
    .reg-table-header {
        border-top: 1px solid #000;
        border-bottom: 1px solid #000;
    }
    .reg-table-last-row {
        border-bottom: 1px solid #000;
    }
    .reg-table-footer {
        font-size: 14px;
        text-align: left;
        margin-top: 10px;
    }
    </style>
    <div class="reg-table-container">
        <div class="reg-table-title">Table 2</div>
        <div class="reg-table-subtitle">Linear Regression Results for {{dep_label}}</div>
        <table class="reg-table">
            <tr class="reg-table-header">
                <th>Variable</th>
                <th>B</th>
                <th>β</th>
                <th>p</th>
                <th>Significance</th>
            </tr>
            {{data_rows}}
        </table>
        <div class="reg-table-footer">
            <i>R</i>² = {{r_squared:.3f}}, Adjusted <i>R</i>² = {{adj_r_squared:.3f}}, df = {{df}}
        </div>
    </div>
    """
    data_rows = ""
    for i, (label, code, coef, std_coef, p) in enumerate(zip(indep_labels, indep_codes, coefs, std_coefs, p_values)):
        row_class = "reg-table-last-row" if i == len(coefs) - 1 else ""
        coef_display = f"{coef:.3f}" if not np.isnan(coef) else "-"
        std_coef_display = "--" if i == 0 else f"{std_coef:.3f}" if not np.isnan(std_coef) else "-"
        p_display = "< .001" if p < 0.001 else f"{p:.3f}" if not np.isnan(p) else "-"
        sig_display = "**" if p < 0.01 else "*" if p < 0.05 else "" if not np.isnan(p) else "-"
        var_display = label
        row = f"""
        <tr class="{row_class}">
            <th>{var_display}</th>
            <td>{coef_display}</td>
            <td>{std_coef_display}</td>
            <td>{p_display}</td>
            <td>{sig_display}</td>
        </tr>
        """
        data_rows += row
    
    full_html = html_content.replace("{{dep_label}}", dep_label).replace("{{dep_code}}", dep_code).replace("{{data_rows}}", data_rows).replace("{{r_squared:.3f}}", f"{r_squared:.3f}").replace("{{adj_r_squared:.3f}}", f"{adj_r_squared:.3f}").replace("{{df}}", str(df))
    return full_html

# Access data from session state
df = st.session_state.get('df', None)
variable_labels = st.session_state.get('variable_labels', {})
value_labels = st.session_state.get('value_labels', {})
visible_columns = st.session_state.get('visible_columns', [])

# Initialize session state for this page
if 'regression_results' not in st.session_state:
    st.session_state.regression_results = None
if 'regression_completed' not in st.session_state:
    st.session_state.regression_completed = False
if 'regression_show_write_up' not in st.session_state:
    st.session_state.regression_show_write_up = False
if 'regression_write_up_content' not in st.session_state:
    st.session_state.regression_write_up_content = None
if 'regression_dep_var' not in st.session_state:
    st.session_state.regression_dep_var = None
if 'regression_indep_vars' not in st.session_state:
    st.session_state.regression_indep_vars = None
if 'regression_covariates' not in st.session_state:
    st.session_state.regression_covariates = None
if 'regression_used_brr' not in st.session_state:
    st.session_state.regression_used_brr = False

# Streamlit UI
st.title("Linear Regression Analysis")

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
        st.warning("At least two numeric variables or domains are required for linear regression.")
    else:
        # Select Dependent Variable (default to MATH domain)
        st.write("Select dependent variable (domain or numeric variable):")
        all_dep_options = domain_options + unique_var_labels
        default_dep_label = "Mathematics score"
        if default_dep_label not in all_dep_options:
            st.warning(f"Default dependent variable '{default_dep_label}' not found in available options: {all_dep_options}. Falling back to first variable.")
            default_index = 0
        else:
            default_index = all_dep_options.index(default_dep_label)
        
        dep_label = st.selectbox(
            "Dependent Variable",
            all_dep_options,
            index=default_index,
            key="dep_var"
        )
        
        # Determine if the dependent variable is a domain or a regular variable
        if dep_label in label_to_domain:
            dep_domain = label_to_domain[dep_label]
            dep_vars = pv_domains[dep_domain]
            dep_is_pv = True
        else:
            dep_vars = [label_to_var[dep_label]]
            dep_is_pv = False
        
        # Select Covariates (default to ST004D01T and ESCS)
        st.write("Select covariates (default: ST004D01T, ESCS):")
        covariate_options = [label for label in unique_var_labels if label != dep_label]
        default_covariates = []
        for col in ["ST004D01T", "ESCS"]:
            label = variable_labels.get(col, col)
            if label in seen_labels and label in covariate_options:
                default_covariates.append(label)
        covariate_labels = st.multiselect(
            "Covariates",
            covariate_options,
            default=default_covariates,
            key="covariates"
        )
        covariates = [label_to_var[label] for label in covariate_labels if label in label_to_var]
        
        # Select Additional Predictor Variables (no defaults, include domains)
        st.write("Select additional predictor variables (optional):")
        indep_options = [
            label for label in (domain_options + unique_var_labels) 
            if label != dep_label and label not in covariate_labels
        ]
        indep_labels = st.multiselect(
            "Predictor Variables",
            indep_options,
            default=[],
            key="indep_vars"
        )
        
        # Split predictors into domains and regular variables
        indep_domains = []
        indep_vars = []
        for label in indep_labels:
            if label in label_to_domain:
                domain = label_to_domain[label]
                indep_domains.append(domain)
            else:
                indep_vars.append(label_to_var[label])
        
        # Combine covariates and additional predictor variables
        all_indep_vars = covariates + indep_vars
        all_indep_labels = covariate_labels + [label for label in indep_labels if label not in label_to_domain]
        for domain in indep_domains:
            all_indep_labels.append(domain_to_label.get(domain, domain))
        
        run_analysis = st.button("Run Analysis", key="run_regression")
        
        # Update session state with selections
        if (dep_label != st.session_state.regression_dep_var or 
            all_indep_vars != st.session_state.regression_indep_vars or
            covariates != st.session_state.regression_covariates):
            st.session_state.regression_dep_var = dep_label
            st.session_state.regression_indep_vars = all_indep_vars
            st.session_state.regression_covariates = covariates
            if not run_analysis:  # Only reset if not running analysis
                st.session_state.regression_results = None
                st.session_state.regression_completed = False
                st.session_state.regression_show_write_up = False
                st.session_state.regression_write_up_content = None
                st.session_state.regression_used_brr = False
        
        if run_analysis and dep_vars and (all_indep_vars or indep_domains):
            try:
                if 'W_FSTUWT' not in df.columns:
                    st.error("Final student weight (W_FSTUWT) not found in the dataset.")
                else:
                    # Check for replicate weights availability
                    replicate_weight_cols = [f"W_FSTURWT{i}" for i in range(1, 81)]
                    missing_weights = [col for col in replicate_weight_cols if col not in df.columns]
                    use_brr = len(missing_weights) == 0  # Use BRR if no replicate weights are missing
                    
                    # Prepare columns for data filtering
                    columns = covariates + indep_vars + ['W_FSTUWT']
                    if use_brr:
                        columns += replicate_weight_cols
                    for domain in indep_domains:
                        columns.extend(pv_domains[domain])
                    if dep_is_pv:
                        columns.extend(dep_vars)
                    else:
                        columns.append(dep_vars[0])
                    
                    reg_data = df[columns].dropna()
                    if len(reg_data) < len(covariates) + len(indep_vars) + len(indep_domains) + 1:
                        st.warning("Insufficient non-missing data for regression (need at least as many observations as parameters).")
                        st.session_state.regression_completed = False
                        st.session_state.regression_show_write_up = False
                        st.session_state.regression_write_up_content = None
                        st.session_state.regression_used_brr = False
                    else:
                        st.write(f"Dataset size after dropping NA: {len(reg_data)} rows")
                        num_pvs = len(dep_vars) if dep_is_pv else 1
                        if indep_domains:
                            num_pvs = max(num_pvs, 10)  # Ensure 10 iterations if any PV domains are involved
                        
                        # Prepare lists to store results across PVs
                        all_coefs = []
                        all_se = []
                        all_r_squared = []
                        all_adj_r_squared = []
                        all_aic = []
                        all_bic = []
                        all_f_stat = []
                        all_f_pvalue = []
                        
                        # Progress bar for PV iterations
                        pv_progress = st.progress(0)
                        total_pv_iterations = num_pvs
                        
                        for pv_idx in range(num_pvs):
                            pv_num = pv_idx + 1
                            st.write(f"Processing plausible value {pv_num}/{num_pvs}...")
                            
                            # Select the PVs for this iteration
                            if dep_is_pv:
                                dep_var = dep_vars[pv_idx]
                            else:
                                dep_var = dep_vars[0]
                            
                            # Prepare predictors
                            X_vars = covariates + indep_vars
                            for domain in indep_domains:
                                pv_var = pv_domains[domain][pv_idx % len(pv_domains[domain])]
                                X_vars.append(pv_var)
                            
                            # Prepare X and y
                            X = reg_data[X_vars].values
                            y = reg_data[dep_var].values
                            w = reg_data['W_FSTUWT'].values
                            
                            # Check for invalid data
                            if np.any(np.isnan(X)) or np.any(np.isnan(y)) or np.any(np.isnan(w)):
                                st.error("Input data contains NaN values after preprocessing.")
                                st.session_state.regression_completed = False
                                st.session_state.regression_show_write_up = False
                                st.session_state.regression_write_up_content = None
                                st.session_state.regression_used_brr = False
                                st.stop()
                            
                            # Log distribution of each predictor
                            st.write(f"Logging distribution of predictors for PV {pv_num}...")
                            for col in X_vars:
                                values = reg_data[col].values
                                unique_vals = np.unique(values[~np.isnan(values)])
                                variance = np.var(values[~np.isnan(values)])
                                st.write(f"Predictor {col}: Unique values = {len(unique_vals)}, Variance = {variance:.4f}")
                            
                            # Unstandardized regression
                            st.write(f"Running unstandardized regression for PV {pv_num}...")
                            results = weighted_linear_regression(X, y, w)
                            if results is None:
                                st.session_state.regression_completed = False
                                st.session_state.regression_show_write_up = False
                                st.session_state.regression_write_up_content = None
                                st.session_state.regression_used_brr = False
                                st.stop()
                            coefs = results.params
                            r_squared = results.rsquared
                            adj_r_squared = results.rsquared_adj
                            aic = results.aic
                            bic = results.bic
                            f_stat = results.fvalue
                            f_pvalue = results.f_pvalue
                            
                            # Compute p-values
                            if use_brr:
                                st.write(f"Calculating standard errors using replicate weights for PV {pv_num}...")
                                brr_progress = st.progress(0)
                                se = compute_brr_se_regression(X, y, replicate_weight_cols, reg_data, brr_progress)
                            else:
                                se = results.bse  # Use statsmodels standard errors as fallback
                            
                            # Store results
                            all_coefs.append(coefs)
                            all_se.append(se)
                            all_r_squared.append(r_squared)
                            all_adj_r_squared.append(adj_r_squared)
                            all_aic.append(aic)
                            all_bic.append(bic)
                            all_f_stat.append(f_stat)
                            all_f_pvalue.append(f_pvalue)
                            
                            # Update PV progress
                            pv_progress.progress((pv_idx + 1) / total_pv_iterations)
                        
                        # Combine results using Rubin's rules
                        st.write("Combining results across plausible values using Rubin's rules...")
                        combined_coefs, combined_se, p_values = apply_rubins_rules(all_coefs, all_se, len(reg_data), len(X_vars) + 1)
                        
                        # Average model fit statistics
                        r_squared = np.mean(all_r_squared)
                        adj_r_squared = np.mean(all_adj_r_squared)
                        aic = np.mean(all_aic)
                        bic = np.mean(all_bic)
                        f_stat = np.mean(all_f_stat)
                        f_pvalue = np.mean(all_f_pvalue)
                        df_reg = len(reg_data) - len(X_vars) - 1
                        
                        # Standardized regression (using combined coefficients for display purposes)
                        st.write("Running standardized regression for display...")
                        X_std = np.array([weighted_standardize(X[:, j], w) for j in range(X.shape[1])]).T
                        y_std = weighted_standardize(reg_data[dep_var].values, w)  # Use last PV for standardization
                        std_results = weighted_linear_regression(X_std, y_std, w)
                        if std_results is None:
                            st.session_state.regression_completed = False
                            st.session_state.regression_show_write_up = False
                            st.session_state.regression_write_up_content = None
                            st.session_state.regression_used_brr = False
                            st.stop()
                        std_coefs = std_results.params
                        
                        # Store results in session state
                        st.write("Storing combined results...")
                        st.session_state.regression_results = {
                            'dep_label': dep_label,
                            'dep_code': dep_var if not dep_is_pv else dep_domain,
                            'indep_labels': ["(Intercept)"] + all_indep_labels,
                            'indep_codes': ["(Intercept)"] + X_vars,
                            'coefs': combined_coefs,
                            'std_coefs': std_coefs,
                            'p_values': p_values,
                            'r_squared': r_squared,
                            'adj_r_squared': adj_r_squared,
                            'aic': aic,
                            'bic': bic,
                            'f_stat': f_stat,
                            'f_pvalue': f_pvalue,
                            'df_reg': df_reg
                        }
                        st.session_state.regression_completed = True
                        st.session_state.regression_show_write_up = False
                        st.session_state.regression_write_up_content = None
                        st.session_state.regression_used_brr = use_brr
                        
                        # Render the regression table
                        st.write("Rendering results table...")
                        table_html = render_regression_table(
                            dep_label, dep_var if not dep_is_pv else dep_domain,
                            ["(Intercept)"] + all_indep_labels, ["(Intercept)"] + X_vars,
                            combined_coefs, std_coefs, p_values, r_squared, adj_r_squared, df_reg
                        )
                        components.html(table_html, height=300, scrolling=True)
                        st.write("Linear regression analysis completed.")
            except Exception as e:
                st.error(f"Error computing linear regression: {str(e)}")
                st.session_state.regression_completed = False
                st.session_state.regression_show_write_up = False
                st.session_state.regression_write_up_content = None
                st.session_state.regression_used_brr = False
        elif st.session_state.regression_results and st.session_state.regression_completed:
            if (st.session_state.regression_dep_var == dep_label and 
                st.session_state.regression_indep_vars == all_indep_vars):
                results = st.session_state.regression_results
                table_html = render_regression_table(
                    results['dep_label'], results['dep_code'], results['indep_labels'],
                    results['indep_codes'], results['coefs'], results['std_coefs'],
                    results['p_values'], results['r_squared'], results['adj_r_squared'], results['df_reg']
                )
                components.html(table_html, height=300, scrolling=True)
                st.write("Linear regression analysis completed.")
            else:
                st.write("Variable selection has changed. Please click 'Run Analysis' to compute the regression with the new variables.")
        elif dep_vars and (all_indep_vars or indep_domains):
            st.write("Please click 'Run Analysis' to compute the linear regression.")
        else:
            st.write("Please select a dependent variable and at least one independent variable (or covariate) to compute the linear regression.")
        
        if st.session_state.regression_results and st.session_state.regression_completed:
            if (st.session_state.regression_dep_var == dep_label and 
                st.session_state.regression_indep_vars == all_indep_vars):
                if st.button("Generate Write-Up", key="writeup_regression"):
                    results = st.session_state.regression_results
                    num_pvs = len(dep_vars) if dep_is_pv else 1
                    if indep_domains:
                        num_pvs = max(num_pvs, 10)  # Ensure 10 iterations if any PV domains are involved
                    write_up = generate_regression_write_up(
                        results['dep_label'], results['dep_code'], results['indep_labels'],
                        results['indep_codes'], results['coefs'], results['p_values'],
                        results['r_squared'], results['adj_r_squared'], results['aic'],
                        results['bic'], results['f_stat'], results['f_pvalue'],
                        results['df_reg'], st.session_state.regression_used_brr, num_pvs
                    )
                    st.session_state.regression_show_write_up = True
                    st.session_state.regression_write_up_content = write_up
        
        # Display the write-up if it exists
        if st.session_state.regression_show_write_up and st.session_state.regression_write_up_content:
            st.markdown("### Sample Write-Up", unsafe_allow_html=True)
            st.markdown(st.session_state.regression_write_up_content, unsafe_allow_html=True)

# Instructions section
st.header("Instructions")
st.markdown("""
- **Select Variables**: Choose a dependent variable (default: Mathematics score), covariates (default: ST004D01T, ESCS), and additional predictor variables (optional) from the dropdown menus. Plausible value domains (e.g., Mathematics, Reading) will use all 10 plausible values for analysis.
- **Run Analysis**: Click "Run Analysis" to perform the weighted linear regression. Analyses involving plausible values will be run 10 times and combined using Rubin's rules.
- **Generate Write-Up**: After running the analysis, click "Generate Write-Up" to see a sample description of the results.
- **Navigate**: Use the sidebar to switch between different analysis types or return to the main page to upload a new dataset.
""")