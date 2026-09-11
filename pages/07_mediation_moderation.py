import streamlit as st
import pandas as pd
import numpy as np
import re
import statsmodels.api as sm
import streamlit.components.v1 as components
from scipy.stats import t, anderson, probplot
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# Add logo that persists across all pages
try:
    st.logo("assets/logo.png")  # Replace with the path to your logo file, e.g., "assets/logo.png"
except Exception as e:
    st.error(f"Failed to load logo: {e}")

# Streamlit app configuration
st.set_page_config(page_title="Moderation - PISA Data Exploration Tool", layout="wide")

# Function to compute weighted standard deviation
def weighted_std(x, w):
    try:
        # Remove NaN values
        mask = ~(np.isnan(x) | np.isnan(w))
        x = x[mask]
        w = w[mask]
        if len(x) < 2 or np.sum(w) == 0:
            return np.nan
        
        # Weighted mean
        w_sum = np.sum(w)
        w_mean = np.sum(w * x) / w_sum
        
        # Weighted variance
        variance = np.sum(w * (x - w_mean)**2) / w_sum
        
        # Weighted standard deviation
        return np.sqrt(variance)
    except Exception as e:
        st.error(f"Error in weighted_std: {str(e)}")
        return np.nan

# Function to compute weighted OLS regression for a single set of variables
def weighted_ols_regression(x, y, w, var_names):
    try:
        # Remove NaN values
        mask = ~(np.isnan(y) | np.isnan(w) | np.any(np.isnan(x), axis=1))
        x = x[mask]
        y = y[mask]
        w = w[mask]
        if len(x) < 2:
            return [(var, np.nan, np.nan, np.nan, np.nan) for var in ['Intercept'] + var_names], np.nan, np.nan, {}
        
        # Compute weighted standard deviations for standardization
        sd_y = weighted_std(y, w)
        sd_x = [weighted_std(x[:, i], w) for i in range(x.shape[1])]
        
        # Validate standard deviations
        if np.isnan(sd_y) or sd_y == 0:
            st.warning("Standard deviation of the dependent variable is zero or NaN. Standardized coefficients (β) cannot be computed.")
        
        for i, sd in enumerate(sd_x):
            if np.isnan(sd) or sd == 0:
                st.warning(f"Standard deviation of predictor {var_names[i]} is zero or NaN. Standardized coefficient (β) for this predictor will be NaN.")
        
        # Add constant for intercept
        X = sm.add_constant(x)
        
        # Fit weighted OLS model
        model = sm.WLS(y, X, weights=w).fit()
        
        # Extract coefficients, standard errors, p-values, and compute standardized coefficients
        results = []
        for idx, var in enumerate(['Intercept'] + var_names):
            coef = model.params[idx]
            se = model.bse[idx]
            p_value = model.pvalues[idx]
            # Compute standardized coefficient (β)
            if idx == 0:  # Intercept
                std_coef = np.nan  # Standardized intercept is not typically meaningful
            else:
                if np.isnan(sd_y) or sd_y == 0 or np.isnan(sd_x[idx-1]) or sd_x[idx-1] == 0:
                    std_coef = np.nan
                else:
                    std_coef = coef * (sd_x[idx-1] / sd_y)
            results.append((var, coef, std_coef, se, p_value))
        
        # Get R-squared and adjusted R-squared values
        r_squared = model.rsquared if hasattr(model, 'rsquared') else np.nan
        r_squared_adj = model.rsquared_adj if hasattr(model, 'rsquared_adj') else np.nan
        
        # Compute model diagnostics
        diagnostics = {}
        
        # F-statistic and p-value
        f_stat = model.fvalue if hasattr(model, 'fvalue') else np.nan
        f_pvalue = model.f_pvalue if hasattr(model, 'f_pvalue') else np.nan
        df_model = model.df_model if hasattr(model, 'df_model') else np.nan
        df_resid = model.df_resid if hasattr(model, 'df_resid') else np.nan
        diagnostics['f_stat'] = (f_stat, f_pvalue, df_model, df_resid)
        
        # Normality of residuals (Anderson-Darling test)
        if len(model.resid) > 3:  # Ensure enough data for the test
            ad_result = anderson(model.resid, dist='norm')
            ad_stat = ad_result.statistic
            # Check if the test statistic exceeds the critical value at 5% significance level
            critical_value_5 = ad_result.critical_values[2]  # Index 2 corresponds to 5% significance
            ad_reject_normality = ad_stat > critical_value_5
        else:
            ad_stat, ad_reject_normality = np.nan, False
        diagnostics['anderson_darling'] = (ad_stat, ad_reject_normality)
        
        # Homoscedasticity (Breusch-Pagan test)
        if len(x) > len(var_names) + 1:  # Ensure enough data for the test
            bp_lm_stat, bp_pvalue, _, _ = het_breuschpagan(model.resid, X)
        else:
            bp_lm_stat, bp_pvalue = np.nan, np.nan
        diagnostics['breusch_pagan'] = (bp_lm_stat, bp_pvalue)
        
        # Multicollinearity (VIF)
        vif_values = []
        if x.shape[1] > 0:  # Ensure there are predictors to compute VIF
            for i in range(x.shape[1]):
                vif = variance_inflation_factor(X, i + 1)  # Skip the intercept (column 0)
                vif_values.append(vif)
        max_vif = max(vif_values) if vif_values else np.nan
        diagnostics['max_vif'] = max_vif
        
        return results, r_squared, r_squared_adj, diagnostics
    except Exception as e:
        st.error(f"Error in weighted_ols_regression: {str(e)}")
        return [(var, np.nan, np.nan, np.nan, np.nan) for var in ['Intercept'] + var_names], np.nan, np.nan, {}

# Function to compute BRR standard errors for regression coefficients
def compute_brr_se_regression(x, y, replicate_weights, reg_data, progress_bar=None, var_names=None):
    if var_names is None:
        var_names = []
    try:
        # Initial regression with final student weights for reference
        main_results, main_r_squared, main_r_squared_adj, main_diagnostics = weighted_ols_regression(x, y, reg_data['W_FSTUWT'].values, var_names)
        main_coefs = [result[1] for result in main_results]  # Unstandardized coefficients
        main_std_coefs = [result[2] for result in main_results]  # Standardized coefficients
        if all(np.isnan(main_coefs)):
            return ([np.nan] * len(main_coefs), [np.nan] * len(main_std_coefs))
        
        # Compute coefficients for each replicate weight
        replicate_coefs = {idx: [] for idx in range(len(main_coefs))}
        replicate_std_coefs = {idx: [] for idx in range(len(main_std_coefs))}
        total_weights = len(replicate_weights)
        for idx, weight_col in enumerate(replicate_weights):
            # Create a DataFrame with only the weights to handle NA dropping
            data = reg_data[[weight_col, 'W_FSTUWT']].copy()
            data = data.dropna()
            if len(data) < 2:
                continue
            
            # Subset x and y to match the non-NA indices of the weights
            indices = data.index
            pos = reg_data.index.get_indexer(indices)
            pos = pos[pos != -1]  # Remove invalid indices
            if len(pos) < 2:
                continue
            x_rep = x[pos]
            y_rep = y[pos]
            w_rep = data[weight_col].values
            
            # Compute regression with replicate weights
            rep_results, _, _, _ = weighted_ols_regression(x_rep, y_rep, w_rep, var_names)
            for coef_idx, (var, coef, std_coef, _, _) in enumerate(rep_results):
                if not np.isnan(coef):
                    replicate_coefs[coef_idx].append(coef)
                if not np.isnan(std_coef):
                    replicate_std_coefs[coef_idx].append(std_coef)
        
        # Compute BRR standard errors for each coefficient (unstandardized and standardized)
        brr_se = []
        brr_se_std = []
        for idx, (var, main_coef, main_std_coef, _, _) in enumerate(main_results):
            # Unstandardized coefficient SE
            if not replicate_coefs[idx]:
                brr_se.append(np.nan)
            else:
                replicate_coefs_array = np.array(replicate_coefs[idx])
                se = np.sqrt((1 / 80) * np.sum((replicate_coefs_array - main_coef) ** 2))
                brr_se.append(se)
            
            # Standardized coefficient SE
            if not replicate_std_coefs[idx]:
                brr_se_std.append(np.nan)
            else:
                replicate_std_coefs_array = np.array(replicate_std_coefs[idx])
                se_std = np.sqrt((1 / 80) * np.sum((replicate_std_coefs_array - main_std_coef) ** 2))
                brr_se_std.append(se_std)
        
        return (brr_se, brr_se_std)
    except Exception as e:
        st.error(f"Error in compute_brr_se_regression: {str(e)}")
        return ([np.nan] * (len(var_names) + 1), [np.nan] * (len(var_names) + 1))

# Function to compute p-values using BRR standard errors
def compute_brr_p_value(coef, se, n):
    try:
        if np.isnan(se) or se == 0 or np.isnan(coef):
            return np.nan
        t_stat = coef / se
        p_value = 2 * (1 - t.cdf(np.abs(t_stat), df=n-2))
        return p_value
    except Exception as e:
        st.error(f"Error in compute_brr_p_value: {str(e)}")
        return np.nan

# Function to apply Rubin's rules for combining regression results across plausible values
def apply_rubins_rules_regression(all_results, n, var_names):
    try:
        # Separate results, R-squared, adjusted R-squared, and diagnostics
        results_list = [result[0] for result in all_results]  # List of (var, coef, std_coef, se, p_value) tuples
        r_squared_list = [result[1] for result in all_results]  # List of R-squared values
        r_squared_adj_list = [result[2] for result in all_results]  # List of adjusted R-squared values
        diagnostics_list = [result[3] for result in all_results]  # List of diagnostics dictionaries
        
        # Transpose results to group by variable: list of lists where each inner list contains results for one variable across PVs
        combined_results = []
        var_results = list(zip(*results_list))  # One list per variable (including Intercept)
        
        for idx, var_result in enumerate(var_results):
            # Determine the variable name
            var = 'Intercept' if idx == 0 else var_names[idx - 1]
            
            coefs_list = [result[1] for result in var_result]  # Unstandardized coefficients
            std_coefs_list = [result[2] for result in var_result]  # Standardized coefficients
            se_list = [result[3] for result in var_result]
            missing_pct_list = [result[5] for result in var_result]  # Missing percentages
            
            # Skip if all coefficients are NaN
            if np.all(np.isnan(coefs_list)):
                combined_results.append((var, np.nan, np.nan, np.nan, np.nan, np.nan))
                continue
            
            num_pvs = len([c for c in coefs_list if not np.isnan(c)])
            if num_pvs == 0:
                combined_results.append((var, np.nan, np.nan, np.nan, np.nan, np.nan))
                continue
            
            # Combined point estimate (unstandardized coefficient)
            combined_coef = np.nanmean(coefs_list)
            
            # Combined point estimate (standardized coefficient)
            # Check if there are any non-NaN standardized coefficients to avoid warning
            std_coefs_array = np.array(std_coefs_list)
            if np.all(np.isnan(std_coefs_array)):
                combined_std_coef = np.nan
            else:
                combined_std_coef = np.nanmean(std_coefs_array)
            
            # Within-imputation variance
            within_var = np.nanmean(np.array(se_list)**2)
            
            # Between-imputation variance (unstandardized coefficient)
            between_var = np.nanvar(coefs_list, ddof=1)
            
            # Total variance (unstandardized coefficient)
            total_var = within_var + (1 + 1/num_pvs) * between_var
            
            # Combined standard error (unstandardized coefficient)
            combined_se = np.sqrt(total_var)
            
            # Compute p-value
            p_value = compute_brr_p_value(combined_coef, combined_se, n)
            
            # Average missing percentage across PVs, avoiding empty slice warning
            non_none_missing_pcts = [pct for pct in missing_pct_list if pct is not None]
            missing_pct = np.nanmean(non_none_missing_pcts) if non_none_missing_pcts else None
            
            combined_results.append((var, combined_coef, combined_std_coef, combined_se, p_value, missing_pct))
        
        # Average R-squared and adjusted R-squared across PVs
        r_squared = np.nanmean(r_squared_list)
        r_squared_adj = np.nanmean(r_squared_adj_list)
        
        # Combine diagnostics across PVs
        combined_diagnostics = {}
        
        # F-statistic: average the statistic and p-value, use the last df values
        f_stats = [d.get('f_stat', (np.nan, np.nan, np.nan, np.nan))[0] for d in diagnostics_list]
        f_pvalues = [d.get('f_stat', (np.nan, np.nan, np.nan, np.nan))[1] for d in diagnostics_list]
        df_model = diagnostics_list[-1].get('f_stat', (np.nan, np.nan, np.nan, np.nan))[2]
        df_resid = diagnostics_list[-1].get('f_stat', (np.nan, np.nan, np.nan, np.nan))[3]
        combined_diagnostics['f_stat'] = (np.nanmean(f_stats), np.nanmean(f_pvalues), df_model, df_resid)
        
        # Anderson-Darling: average the statistic, combine rejection decisions with logical OR
        ad_stats = [d.get('anderson_darling', (np.nan, False))[0] for d in diagnostics_list]
        ad_reject = any(d.get('anderson_darling', (np.nan, False))[1] for d in diagnostics_list)
        combined_diagnostics['anderson_darling'] = (np.nanmean(ad_stats), ad_reject)
        
        # Breusch-Pagan: average the statistic and p-value
        bp_lm_stats = [d.get('breusch_pagan', (np.nan, np.nan))[0] for d in diagnostics_list]
        bp_pvalues = [d.get('breusch_pagan', (np.nan, np.nan))[1] for d in diagnostics_list]
        combined_diagnostics['breusch_pagan'] = (np.nanmean(bp_lm_stats), np.nanmean(bp_pvalues))
        
        # Max VIF: take the maximum across PVs
        max_vifs = [d.get('max_vif', np.nan) for d in diagnostics_list]
        combined_diagnostics['max_vif'] = np.nanmax(max_vifs)
        
        return combined_results, r_squared, r_squared_adj, combined_diagnostics
    except Exception as e:
        st.error(f"Error in apply_rubins_rules_regression: {str(e)}")
        return [(var, np.nan, np.nan, np.nan, np.nan, np.nan) for var in ['Intercept'] + var_names], np.nan, np.nan, {}

# Function to plot to base64
def plot_to_base64():
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _wls_cov(work_df, dep_col, independent_vars, weight_col="W_FSTUWT"):
    cols = [dep_col] + list(independent_vars) + [weight_col]
    data = work_df[cols].dropna()
    if len(data) < len(independent_vars) + 3:
        return None, None, len(data)
    y = data[dep_col].astype(float).values
    X = sm.add_constant(data[independent_vars].astype(float).values)
    w = data[weight_col].astype(float).values
    model = sm.WLS(y, X, weights=w).fit()
    return model.cov_params(), model.df_resid, len(data)


def compute_simple_slopes(work_df, dep_col, independent_vars, results, selected_pairs, label_to_var, center_ix):
    """Pick-a-point simple slopes for each A × B term. SEs use WLS covariance (approx. if DV is a PV)."""
    outputs = []
    cov, df_resid, n = _wls_cov(work_df, dep_col, independent_vars)
    coef_by_code = {}
    for i, code in enumerate(independent_vars):
        if i + 1 < len(results):
            coef_by_code[code] = results[i + 1][1]
    intercept = results[0][1] if results else np.nan

    for a_lab, b_lab in selected_pairs:
        a_code = label_to_var.get(a_lab)
        b_code = label_to_var.get(b_lab)
        ix_code = f"IX_{a_code}__{b_code}"
        if a_code not in coef_by_code or b_code not in coef_by_code or ix_code not in coef_by_code:
            outputs.append({"title": f"{a_lab} × {b_lab}", "error": "Could not match interaction coefficients."})
            continue
        b_a = coef_by_code[a_code]
        b_b = coef_by_code[b_code]
        b_ab = coef_by_code[ix_code]
        a_raw = work_df[a_code].astype(float)
        b_raw = work_df[b_code].astype(float)
        mean_a, mean_b = a_raw.mean(), b_raw.mean()
        sd_a, sd_b = a_raw.std(ddof=1), b_raw.std(ddof=1)

        def slope_se(focal_idx, int_idx, z):
            if cov is None:
                return np.nan, np.nan
            v_f = cov[focal_idx, focal_idx]
            v_i = cov[int_idx, int_idx]
            c_fi = cov[focal_idx, int_idx]
            var = v_f + (z ** 2) * v_i + 2 * z * c_fi
            se = np.sqrt(var) if var > 0 else np.nan
            return se, df_resid

        def rows_for(focal_lab, foc_code, mod_lab, mod_code, b_foc, mean_m, sd_m, raw_m):
            idx_f = independent_vars.index(foc_code) + 1
            idx_i = independent_vars.index(ix_code) + 1
            nuniq = raw_m.nunique(dropna=True)
            if nuniq <= 2:
                levels = []
                for val in sorted(raw_m.dropna().unique()):
                    z = val  # product used raw binary values (not centered)
                    levels.append((f"{mod_lab} = {val:g}", z, b_foc + b_ab * z))
            else:
                # Product used (A-meanA)*(B-meanB) when centering is on
                zs = [(-sd_m if center_ix else mean_m - sd_m),
                      (0.0 if center_ix else mean_m),
                      (sd_m if center_ix else mean_m + sd_m)]
                labels_lvl = [
                    f"Low {mod_lab} (−1 SD)",
                    f"Mean {mod_lab}",
                    f"High {mod_lab} (+1 SD)",
                ]
                levels = []
                for lab, z in zip(labels_lvl, zs):
                    levels.append((lab, z, b_foc + b_ab * z))
            table_rows = []
            for lab, z, slope in levels:
                se, dfr = slope_se(idx_f, idx_i, z)
                if se is not None and not np.isnan(se) and se > 0 and dfr and dfr > 0:
                    t_stat = slope / se
                    p_val = 2 * (1 - t.cdf(np.abs(t_stat), df=dfr))
                else:
                    se, t_stat, p_val = np.nan, np.nan, np.nan
                table_rows.append((lab, slope, se, p_val))
            return table_rows

        rows_a_at_b = rows_for(a_lab, a_code, b_lab, b_code, b_a, mean_b, sd_b, b_raw)
        rows_b_at_a = rows_for(b_lab, b_code, a_lab, a_code, b_b, mean_a, sd_a, a_raw)

        # Predicted lines: Y vs A at low/mean/high B, others at mean
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        a_grid = np.linspace(a_raw.quantile(0.05), a_raw.quantile(0.95), 40)
        other_means = {}
        for code in independent_vars:
            if code not in (a_code, b_code, ix_code):
                other_means[code] = work_df[code].astype(float).mean()
        if b_raw.nunique(dropna=True) <= 2:
            b_levels = [(f"{b_lab} = {v:g}", v) for v in sorted(b_raw.dropna().unique())]
        else:
            b_levels = [
                (f"Low {b_lab} (−1 SD)", mean_b - sd_b),
                (f"Mean {b_lab}", mean_b),
                (f"High {b_lab} (+1 SD)", mean_b + sd_b),
            ]
        for lvl_lab, b_val in b_levels:
            if center_ix and b_raw.nunique(dropna=True) > 2:
                prod = (a_grid - mean_a) * (b_val - mean_b)
            else:
                prod = a_grid * b_val
            yhat = intercept + b_a * a_grid + b_b * b_val + b_ab * prod
            for code, mu in other_means.items():
                yhat = yhat + coef_by_code.get(code, 0.0) * mu
            ax.plot(a_grid, yhat, label=lvl_lab)
        ax.set_xlabel(a_lab)
        ax.set_ylabel("Predicted outcome")
        ax.set_title(f"Simple slopes: {a_lab} at levels of {b_lab}")
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plot_b64 = plot_to_base64()

        outputs.append({
            "title": f"{a_lab} × {b_lab}",
            "focal_a": a_lab,
            "mod_b": b_lab,
            "rows_a_at_b": rows_a_at_b,
            "rows_b_at_a": rows_b_at_a,
            "plot": plot_b64,
            "n": n,
        })
    return outputs


def render_simple_slope_table(title, focal, moderator, rows):
    html = f"""
    <div style="font-family: Times New Roman, Times, serif; margin: 12px 0 20px 0;">
      <div style="font-weight:bold;">Simple slopes of {focal} at levels of {moderator}</div>
      <div style="font-style:italic; margin-bottom:8px;">{title}</div>
      <table style="border-collapse:collapse; font-size:14px;">
        <tr style="border-top:1px solid #000; border-bottom:1px solid #000;">
          <th style="text-align:left; padding:6px 12px; font-weight:normal;">Level of {moderator}</th>
          <th style="padding:6px 12px; font-weight:normal;"><i>B</i></th>
          <th style="padding:6px 12px; font-weight:normal;"><i>SE</i></th>
          <th style="padding:6px 12px; font-weight:normal;"><i>p</i></th>
        </tr>
    """
    for i, (lab, slope, se, p_val) in enumerate(rows):
        slope_d = f"{slope:.2f}" if not np.isnan(slope) else "-"
        se_d = f"{se:.2f}" if not np.isnan(se) else "-"
        if np.isnan(p_val):
            p_d, sig = "-", ""
        elif p_val < 0.001:
            p_d, sig = "&lt; .001", "**"
        elif p_val < 0.01:
            p_d, sig = f"{p_val:.3f}", "*"
        elif p_val < 0.05:
            p_d, sig = f"{p_val:.3f}", "*"
        else:
            p_d, sig = f"{p_val:.3f}", ""
        border = "border-bottom:1px solid #000;" if i == len(rows) - 1 else ""
        html += f"<tr style='{border}'><td style='padding:6px 12px;'>{lab}</td><td style='text-align:center; padding:6px 12px;'>{slope_d}{sig}</td><td style='text-align:center; padding:6px 12px;'>{se_d}</td><td style='text-align:center; padding:6px 12px;'>{p_d}</td></tr>"
    html += """</table>
      <div style="font-size:13px; margin-top:6px;"><i>Note.</i> Continuous moderators evaluated at −1 SD, mean, and +1 SD.
      Standard errors from the weighted OLS covariance matrix (approximate when the outcome is a plausible-value score).
      *<i>p</i> &lt; .05. **<i>p</i> &lt; .01.</div>
    </div>
    """
    return html

# Function to compute linear regression with PVs and BRR
def compute_linear_regression_with_pvs(df, dependent_var, independent_vars, weights, replicate_weights, use_brr, var_labels, status_placeholder=None):
    try:
        # Determine if dependent or independent variables are PV domains
        dep_is_pv = dependent_var.startswith("PV")
        indep_is_pv = [var.startswith("PV") for var in independent_vars]
        
        # Map independent variable codes to their labels
        code_to_label = {code: label for label, code in var_labels.items()}
        indep_var_labels = []
        for var in independent_vars:
            if var.startswith("PV"):
                # For PV domains, use the domain label (e.g., "Mathematics score")
                pv_match = pv_pattern.match(var)
                domain = pv_match.group(2).upper()
                domain_label = domain_to_label.get(domain, domain)
                indep_var_labels.append(domain_label)
            else:
                # For regular variables, use the label from code_to_label
                indep_var_labels.append(code_to_label.get(var, var))
        
        # Initialize dictionary to store missing data percentages
        missing_percentages = {}
        
        # Store original dataset size
        original_size = len(df)
        
        # If neither dependent nor independent vars are PVs, run a single regression
        if not dep_is_pv and not any(indep_is_pv):
            # Subset data and drop rows with missing values
            data = df[[dependent_var] + independent_vars + ['W_FSTUWT'] + replicate_weights].copy()
            total_rows = len(data)
            data = data.dropna()
            final_size = len(data)  # Final sample size after listwise deletion
            if len(data) < 2:
                raise ValueError("Insufficient non-missing data for regression.")
            
            # Calculate missing data percentages for each variable
            missing_dep = (total_rows - len(df[dependent_var].dropna())) / total_rows * 100
            missing_percentages[dependent_var] = missing_dep
            for var in independent_vars:
                missing_var = (total_rows - len(df[var].dropna())) / total_rows * 100
                missing_percentages[var] = missing_var
            
            X = data[independent_vars].values
            y = data[dependent_var].values
            w = data['W_FSTUWT'].values
            
            # Run regression
            results, r_squared, r_squared_adj, diagnostics = weighted_ols_regression(X, y, w, independent_vars)
            if use_brr:
                if status_placeholder:
                    status_placeholder.write("Calculating standard errors using replicate weights...")
                brr_se, brr_se_std = compute_brr_se_regression(X, y, replicate_weights, data, None, independent_vars)
                # Update results with BRR standard errors and recompute p-values
                updated_results = []
                for idx, (var, coef, std_coef, _, _) in enumerate(results):
                    se = brr_se[idx]
                    p_value = compute_brr_p_value(coef, se, len(data))
                    # Replace variable code with label and add missing percentage
                    if idx == 0:  # Intercept
                        updated_var = var
                        missing_pct = None  # No missing percentage for Intercept
                    else:
                        updated_var = indep_var_labels[idx - 1]
                        var_code = independent_vars[idx - 1]
                        missing_pct = missing_percentages.get(var_code, 0.0)
                    updated_results.append((updated_var, coef, std_coef, se, p_value, missing_pct))
                results = updated_results
            else:
                # Replace variable codes with labels in the results and add missing percentage
                updated_results = []
                for idx, (var, coef, std_coef, se, p_value) in enumerate(results):
                    if idx == 0:  # Intercept
                        updated_var = var
                        missing_pct = None
                    else:
                        updated_var = indep_var_labels[idx - 1]
                        var_code = independent_vars[idx - 1]
                        missing_pct = missing_percentages.get(var_code, 0.0)
                    updated_results.append((updated_var, coef, std_coef, se, p_value, missing_pct))
                results = updated_results
            
            # Generate visualizations using the final model
            visualizations = {}
            X = sm.add_constant(X)
            fitted_values = np.dot(X, [r[1] for r in results])  # Compute fitted values using combined coefficients
            residuals = y - fitted_values
            
            # Q-Q Plot
            plt.figure(figsize=(6, 4))
            probplot(residuals, dist="norm", plot=plt)
            plt.title("Q-Q Plot of Residuals")
            visualizations['qq_plot'] = plot_to_base64()
            
            # Residuals vs. Fitted Values Plot
            plt.figure(figsize=(6, 4))
            plt.scatter(fitted_values, residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel("Fitted Values")
            plt.ylabel("Residuals")
            plt.title("Residuals vs. Fitted Values")
            visualizations['resid_vs_fitted'] = plot_to_base64()
            
            # Residuals Histogram
            plt.figure(figsize=(6, 4))
            sns.histplot(residuals, kde=True, stat="density")
            plt.xlabel("Residuals")
            plt.title("Histogram of Residuals")
            visualizations['resid_histogram'] = plot_to_base64()
            
            return results, r_squared, r_squared_adj, diagnostics, visualizations, final_size, original_size
        
        # If either dependent or independent vars are PVs, handle PV analysis
        pv_domain = None
        if dep_is_pv:
            pv_match = pv_pattern.match(dependent_var)
            pv_domain = pv_match.group(2).upper()
            pv_nums = list(range(1, 11))  # PV1 to PV10
            pv_vars = [f"PV{i}{pv_domain}" for i in pv_nums]
        else:
            # Check if any independent variable is a PV
            for var in independent_vars:
                pv_match = pv_pattern.match(var)
                if pv_match:
                    pv_domain = pv_match.group(2).upper()
                    pv_nums = list(range(1, 11))
                    pv_vars = [f"PV{i}{pv_domain}" for i in pv_nums]
                    break
        
        if not pv_domain:
            raise ValueError("No plausible value domain identified.")
        
        # Progress bar for PV iterations
        pv_progress = st.progress(0)
        total_iterations = len(pv_vars)
        iteration_count = 0
        
        all_results = []
        final_size = 0  # Initialize final_size for PV case
        for pv_idx in pv_nums:
            pv_var = f"PV{pv_idx}{pv_domain}"
            if pv_var not in df.columns:
                iteration_count += 1
                pv_progress.progress(iteration_count / total_iterations)
                continue
            
            if status_placeholder:
                status_placeholder.write(f"Processing regression with {pv_var}...")
            
            # Prepare variables for this PV iteration
            if dep_is_pv:
                dep_var = pv_var
                indep_vars = independent_vars
            else:
                dep_var = dependent_var
                indep_vars = []
                for var in independent_vars:
                    if var.startswith("PV"):
                        pv_match = pv_pattern.match(var)
                        if pv_match.group(2).upper() == pv_domain:
                            indep_vars.append(f"PV{pv_idx}{pv_domain}")
                        else:
                            indep_vars.append(var)
                    else:
                        indep_vars.append(var)
            
            # Create a subset DataFrame for this iteration
            columns = [dep_var] + indep_vars + ['W_FSTUWT']
            if use_brr:
                columns += replicate_weights
            data = df[columns].copy()
            total_rows = len(data)
            data = data.dropna()
            final_size = len(data)  # Final sample size after listwise deletion
            if len(data) < 2:
                st.warning(f"Insufficient non-missing data for regression with {pv_var}.")
                iteration_count += 1
                pv_progress.progress(iteration_count / total_iterations)
                continue
            
            # Calculate missing data percentages for each variable
            missing_dep = (total_rows - len(df[dep_var].dropna())) / total_rows * 100
            missing_percentages[dep_var] = missing_dep
            for var in indep_vars:
                missing_var = (total_rows - len(df[var].dropna())) / total_rows * 100
                missing_percentages[var] = missing_var
            
            X = data[indep_vars].values
            y = data[dep_var].values
            w = data['W_FSTUWT'].values
            
            # Run regression
            results, r_squared, r_squared_adj, diagnostics = weighted_ols_regression(X, y, w, indep_vars)
            if use_brr:
                if status_placeholder:
                    status_placeholder.write("Calculating standard errors using replicate weights...")
                brr_se, brr_se_std = compute_brr_se_regression(X, y, replicate_weights, data, None, indep_vars)
                # Update results with BRR standard errors and recompute p-values
                updated_results = []
                for idx, (var, coef, std_coef, _, _) in enumerate(results):
                    se = brr_se[idx]
                    p_value = compute_brr_p_value(coef, se, len(data))
                    # Replace variable code with label and add missing percentage
                    if idx == 0:  # Intercept
                        updated_var = var
                        missing_pct = None  # No missing percentage for Intercept
                    else:
                        updated_var = indep_var_labels[idx - 1]
                        var_code = indep_vars[idx - 1]
                        missing_pct = missing_percentages.get(var_code, 0.0)
                    updated_results.append((updated_var, coef, std_coef, se, p_value, missing_pct))
                results = updated_results
            else:
                # Replace variable codes with labels in the results and add missing percentage
                updated_results = []
                for idx, (var, coef, std_coef, se, p_value) in enumerate(results):
                    if idx == 0:  # Intercept
                        updated_var = var
                        missing_pct = None
                    else:
                        updated_var = indep_var_labels[idx - 1]
                        var_code = indep_vars[idx - 1]
                        missing_pct = missing_percentages.get(var_code, 0.0)
                    updated_results.append((updated_var, coef, std_coef, se, p_value, missing_pct))
                results = updated_results
            
            all_results.append((results, r_squared, r_squared_adj, diagnostics))
            
            iteration_count += 1
            pv_progress.progress(iteration_count / total_iterations)
        
        # Combine results using Rubin's rules
        if not all_results:
            raise ValueError("No valid regression results computed for any plausible values.")
        
        if status_placeholder:
            status_placeholder.write("Combining results across plausible values...")
        combined_results, combined_r_squared, combined_r_squared_adj, combined_diagnostics = apply_rubins_rules_regression(all_results, len(data), indep_var_labels)
        
        # Generate visualizations using the final combined model
        visualizations = {}
        # Recompute residuals and fitted values using the combined coefficients
        data = df[[dependent_var] + independent_vars + ['W_FSTUWT']].dropna()
        X = data[independent_vars].values
        y = data[dependent_var].values
        w = data['W_FSTUWT'].values
        X = sm.add_constant(X)
        fitted_values = np.dot(X, [r[1] for r in combined_results])
        residuals = y - fitted_values
        
        # Q-Q Plot
        plt.figure(figsize=(6, 4))
        probplot(residuals, dist="norm", plot=plt)
        plt.title("Q-Q Plot of Residuals")
        visualizations['qq_plot'] = plot_to_base64()
        
        # Residuals vs. Fitted Values Plot
        plt.figure(figsize=(6, 4))
        plt.scatter(fitted_values, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel("Fitted Values")
        plt.ylabel("Residuals")
        plt.title("Residuals vs. Fitted Values")
        visualizations['resid_vs_fitted'] = plot_to_base64()
        
        # Residuals Histogram
        plt.figure(figsize=(6, 4))
        sns.histplot(residuals, kde=True, stat="density")
        plt.xlabel("Residuals")
        plt.title("Histogram of Residuals")
        visualizations['resid_histogram'] = plot_to_base64()
        
        return combined_results, combined_r_squared, combined_r_squared_adj, combined_diagnostics, visualizations, final_size, original_size
    except Exception as e:
        st.error(f"Error in compute_linear_regression_with_pvs: {str(e)}")
        return [(var, np.nan, np.nan, np.nan, np.nan, None) for var in ['Intercept'] + independent_vars], np.nan, np.nan, {}, {}, 0, 0

# Function to render regression table as HTML
def render_regression_table(dependent_var_label, results, r_squared, r_squared_adj, diagnostics, final_size, original_size):
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
        text-align: center;
        font-weight: normal;  /* Remove bold styling */
    }
    .reg-table th:first-child, .reg-table td:first-child {
        width: 200px !important;
        text-align: left;
        white-space: normal;
        overflow-wrap: break-word;
    }
    .reg-table th:not(:first-child), .reg-table td:not(:first-child) {
        width: 100px !important;
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
    .reg-table-note {
        font-size: 14px;
        text-align: left;
        margin-top: 5px;
    }
    </style>
    <div class="reg-table-container">
        <div class="reg-table-title">Table 1</div>
        <div class="reg-table-subtitle">Weighted Moderation Results for Dependent Variable: {{dependent_var}}</div>
        <table class="reg-table">
            <tr class="reg-table-header">
                <th>Variable</th>
                <th><i>B</i></th>
                <th><i>β</i></th>
                <th><i>SE</i></th>
                <th><i>p</i></th>
                <th>% Missing</th>
            </tr>
            {{data_rows}}
        </table>
        <div class="reg-table-note"><i>Note.</i> <i>R²</i> = {{r_squared}}, Adjusted <i>R²</i> = {{r_squared_adj}}</div>
        <div class="reg-table-note">Model: <i>F</i>({{df_model}}, {{df_resid}}) = {{f_stat}}, <i>p</i> = {{f_pvalue}}</div>
        <div class="reg-table-note">Assumptions: Anderson-Darling: <i>A²</i> = {{ad_stat}}, Normality Rejected at 5% = {{ad_reject}}; Breusch-Pagan: <i>LM</i> = {{bp_lm_stat}}, <i>p</i> = {{bp_pvalue}}; Max <i>VIF</i> = {{max_vif}}</div>
        <div class="reg-table-note">Sample Size: Final N = {{final_size}} ({{percent_retained}}% of original N = {{original_size}} after listwise deletion)</div>
    </div>
    """
    data_rows = ""
    for idx, (var, coef, std_coef, se, p_value, missing_pct) in enumerate(results):
        coef_display = f"{coef:.2f}" if not np.isnan(coef) else "-"
        std_coef_display = f"{std_coef:.2f}" if not np.isnan(std_coef) else "-"
        se_display = f"{se:.2f}" if not np.isnan(se) else "-"
        p_display = "< .001" if p_value < 0.001 else f"{p_value:.2f}" if not np.isnan(p_value) else "-"
        sig_display = "**" if p_value < 0.01 else "*" if p_value < 0.05 else "" if not np.isnan(p_value) else ""
        missing_display = f"{missing_pct:.1f}" if missing_pct is not None and not np.isnan(missing_pct) else "-"
        row_class = "reg-table-last-row" if idx == len(results) - 1 else ""
        row = f"""
        <tr class="{row_class}">
            <th>{var}</th>
            <td>{coef_display}{sig_display}</td>
            <td>{std_coef_display}</td>
            <td>{se_display}</td>
            <td>{p_display}</td>
            <td>{missing_display}</td>
        </tr>
        """
        data_rows += row
    
    r_squared_display = f"{r_squared:.3f}" if not np.isnan(r_squared) else "-"
    r_squared_adj_display = f"{r_squared_adj:.3f}" if not np.isnan(r_squared_adj) else "-"
    
    # Extract diagnostics
    f_stat, f_pvalue, df_model, df_resid = diagnostics.get('f_stat', (np.nan, np.nan, np.nan, np.nan))
    f_stat_display = f"{f_stat:.2f}" if not np.isnan(f_stat) else "-"
    f_pvalue_display = "< .001" if f_pvalue < 0.001 else f"{f_pvalue:.3f}" if not np.isnan(f_pvalue) else "-"
    df_model_display = f"{int(df_model)}" if not np.isnan(df_model) else "-"
    df_resid_display = f"{int(df_resid)}" if not np.isnan(df_resid) else "-"
    
    ad_stat, ad_reject = diagnostics.get('anderson_darling', (np.nan, False))
    ad_stat_display = f"{ad_stat:.2f}" if not np.isnan(ad_stat) else "-"
    ad_reject_display = "Yes" if ad_reject else "No"
    
    bp_lm_stat, bp_pvalue = diagnostics.get('breusch_pagan', (np.nan, np.nan))
    bp_lm_stat_display = f"{bp_lm_stat:.2f}" if not np.isnan(bp_lm_stat) else "-"
    bp_pvalue_display = "< .001" if bp_pvalue < 0.001 else f"{bp_pvalue:.3f}" if not np.isnan(bp_pvalue) else "-"
    
    max_vif = diagnostics.get('max_vif', np.nan)
    max_vif_display = f"{max_vif:.2f}" if not np.isnan(max_vif) else "-"
    
    # Calculate percentage of original dataset retained
    percent_retained = (final_size / original_size * 100) if original_size > 0 else 0
    percent_retained_display = f"{percent_retained:.1f}"
    
    full_html = html_content.replace("{{dependent_var}}", dependent_var_label).replace("{{data_rows}}", data_rows).replace("{{r_squared}}", r_squared_display).replace("{{r_squared_adj}}", r_squared_adj_display).replace("{{f_stat}}", f_stat_display).replace("{{f_pvalue}}", f_pvalue_display).replace("{{df_model}}", df_model_display).replace("{{df_resid}}", df_resid_display).replace("{{ad_stat}}", ad_stat_display).replace("{{ad_reject}}", ad_reject_display).replace("{{bp_lm_stat}}", bp_lm_stat_display).replace("{{bp_pvalue}}", bp_pvalue_display).replace("{{max_vif}}", max_vif_display).replace("{{final_size}}", str(final_size)).replace("{{original_size}}", str(original_size)).replace("{{percent_retained}}", percent_retained_display)
    
    return full_html

# Access data from session state
df = st.session_state.get('df', None)
variable_labels = st.session_state.get('variable_labels', {})
value_labels = st.session_state.get('value_labels', {})
visible_columns = st.session_state.get('visible_columns', [])

if "med_completed" not in st.session_state:
    st.session_state.med_completed = False

# Streamlit UI
st.title("Mediation and moderated mediation")
label = st.session_state.get("dataset_label")
if label:
    st.info(f"Dataset: {label}")
st.caption(
    "Mediation: does M carry part of X → Y? "
    "Moderated mediation: does that indirect effect change across levels of W? "
    "Y may be a PISA score domain; X, M and W should be observed variables. "
    "Single-level weighted OLS, not SEM."
)
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
    
    # Combine domain options and regular numeric variables for the dropdown
    all_var_options = domain_options + unique_var_labels
    
    if len(unique_var_labels) < 2:
        st.warning("Need at least two observed variables plus an outcome for mediation.")
    else:
        st.write("Select the outcome (Y):")
        dependent_var_label = st.selectbox(
            "Outcome (Y)",
            [""] + all_var_options,
            index=0,
            key="med_y",
        )
        if not dependent_var_label:
            dependent_var = None
        elif dependent_var_label in label_to_domain:
            dependent_var = pv_domains[label_to_domain[dependent_var_label]][0]
        else:
            dependent_var = label_to_var[dependent_var_label]

        xm_options = [lab for lab in unique_var_labels if lab != dependent_var_label]
        st.write("Select the predictor (X):")
        x_label = st.selectbox("Predictor (X)", [""] + xm_options, key="med_x")
        st.write("Select the mediator (M) — the variable that may carry the X → Y association:")
        m_label = st.selectbox(
            "Mediator (M)",
            [""] + [lab for lab in xm_options if lab != x_label],
            key="med_m",
        )
        st.write("Optional moderator (W) — leave blank for ordinary mediation:")
        w_label = st.selectbox(
            "Moderator (W)",
            [""] + [lab for lab in xm_options if lab not in (x_label, m_label)],
            key="med_w",
        )
        which_paths = None
        if w_label:
            which_paths = st.radio(
                "W moderates which path?",
                ["a only (X → M)", "b only (M → Y)", "both a and b"],
                index=2,
                key="med_which_paths",
                help="a only = X×W in the mediator model. b only = M×W in the outcome model. both = Hayes-style moderated mediation.",
            )
            st.checkbox("Center continuous variables before products", value=True, key="med_center")
        center_ix = st.session_state.get("med_center", True)

        covariate_options = [
            lab for lab in unique_var_labels
            if lab not in (dependent_var_label, x_label, m_label, w_label) and lab
        ]
        default_covariates = []
        student_gender_label = variable_labels.get("ST004D01T", "ST004D01T")
        for lab in unique_var_labels:
            if lab == student_gender_label or lab.startswith(f"{student_gender_label} ("):
                student_gender_label = lab
                break
        escs_label = variable_labels.get("ESCS", "ESCS")
        for lab in unique_var_labels:
            if lab == escs_label or lab.startswith(f"{escs_label} ("):
                escs_label = lab
                break
        if student_gender_label in covariate_options and student_gender_label not in (x_label, m_label):
            default_covariates.append(student_gender_label)
        if escs_label in covariate_options and escs_label not in (x_label, m_label):
            default_covariates.append(escs_label)
        covariate_labels = st.multiselect(
            "Covariates (optional)",
            covariate_options,
            default=default_covariates,
            key="med_covariates",
        )
        covariates = [label_to_var[lab] for lab in covariate_labels if lab in label_to_var]

        ready = bool(dependent_var and x_label and m_label and x_label in label_to_var and m_label in label_to_var)
        if ready:
            st.success(f"X = {x_label}  →  M = {m_label}  →  Y = {dependent_var_label}")
        else:
            st.info("Choose Y, X and M to estimate mediation.")

        run_analysis = st.button("Run mediation", key="run_mediation")
        
        if run_analysis and ready:
            try:
                if 'W_FSTUWT' not in df.columns:
                    st.error("Final student weight (W_FSTUWT) not found in the dataset.")
                else:
                    # Check for replicate weights availability
                    replicate_weight_cols = [f"W_FSTURWT{i}" for i in range(1, 81)]
                    missing_weights = [col for col in replicate_weight_cols if col not in df.columns]
                    use_brr = len(missing_weights) == 0
                    
                    # Create a placeholder for status messages
                    status_placeholder = st.empty()

                    x_code = label_to_var[x_label]
                    m_code = label_to_var[m_label]
                    w_code = label_to_var[w_label] if w_label else None
                    work_df = df.copy()
                    work_labels = dict(label_to_var)
                    mod_a = bool(w_code) and which_paths in ("a only (X → M)", "both a and b")
                    mod_b = bool(w_code) and which_paths in ("b only (M → Y)", "both a and b")
                    xw_label = mw_label = None
                    w_mean = w_sd = np.nan
                    if w_code:
                        w_raw = work_df[w_code].astype(float)
                        w_mean, w_sd = w_raw.mean(), w_raw.std(ddof=1)
                        x_s = work_df[x_code].astype(float)
                        m_s = work_df[m_code].astype(float)
                        w_s = w_raw.copy()
                        if center_ix:
                            if x_s.nunique(dropna=True) > 2:
                                x_s = x_s - x_s.mean()
                            if m_s.nunique(dropna=True) > 2:
                                m_s = m_s - m_s.mean()
                            if w_s.nunique(dropna=True) > 2:
                                w_s = w_s - w_s.mean()
                        if mod_a:
                            xw_code = f"IX_{x_code}__{w_code}"
                            xw_label = f"{x_label} × {w_label}"
                            work_df[xw_code] = x_s * w_s
                            work_labels[xw_label] = xw_code
                        if mod_b:
                            mw_code = f"IX_{m_code}__{w_code}"
                            mw_label = f"{m_label} × {w_label}"
                            work_df[mw_code] = m_s * w_s
                            work_labels[mw_label] = mw_code

                    def coef_row(results_list, name):
                        for row in results_list:
                            if row[0] == name:
                                return row
                        return None

                    status_placeholder.write("Path a: M ~ X + covariates")
                    a_preds = list(covariates) + [x_code]
                    if w_code:
                        a_preds.append(w_code)
                    if mod_a:
                        a_preds.append(xw_code)
                    res_a, r2_a, r2a_a, diag_a, _, n_a, n0_a = compute_linear_regression_with_pvs(
                        work_df, m_code, a_preds, work_df["W_FSTUWT"], replicate_weight_cols, use_brr, work_labels, status_placeholder
                    )
                    row_a = coef_row(res_a, x_label)
                    row_xw = coef_row(res_a, xw_label) if xw_label else None

                    status_placeholder.write("Path b and c': Y ~ X + M + covariates")
                    b_preds = list(covariates) + [x_code, m_code]
                    if w_code:
                        b_preds.append(w_code)
                    if mod_b:
                        b_preds.append(mw_code)
                    res_b, r2_b, r2a_b, diag_b, viz_b, n_b, n0_b = compute_linear_regression_with_pvs(
                        work_df, dependent_var, b_preds, work_df["W_FSTUWT"], replicate_weight_cols, use_brr, work_labels, status_placeholder
                    )
                    row_b = coef_row(res_b, m_label)
                    row_cp = coef_row(res_b, x_label)
                    row_mw = coef_row(res_b, mw_label) if mw_label else None

                    status_placeholder.write("Path c (total): Y ~ X + covariates")
                    c_preds = list(covariates) + [x_code]
                    if w_code:
                        c_preds.append(w_code)
                    res_c, r2_c, r2a_c, diag_c, _, n_c, n0_c = compute_linear_regression_with_pvs(
                        work_df, dependent_var, c_preds, work_df["W_FSTUWT"], replicate_weight_cols, use_brr, work_labels, status_placeholder
                    )
                    row_c = coef_row(res_c, x_label)

                    a_hat = row_a[1] if row_a else np.nan
                    se_a = row_a[3] if row_a else np.nan
                    p_a = row_a[4] if row_a else np.nan
                    b_hat = row_b[1] if row_b else np.nan
                    se_b = row_b[3] if row_b else np.nan
                    p_b = row_b[4] if row_b else np.nan
                    c_hat = row_c[1] if row_c else np.nan
                    se_c = row_c[3] if row_c else np.nan
                    p_c = row_c[4] if row_c else np.nan
                    cp_hat = row_cp[1] if row_cp else np.nan
                    se_cp = row_cp[3] if row_cp else np.nan
                    p_cp = row_cp[4] if row_cp else np.nan
                    indirect = a_hat * b_hat if not (np.isnan(a_hat) or np.isnan(b_hat)) else np.nan
                    if not any(np.isnan(v) for v in (a_hat, b_hat, se_a, se_b)):
                        sobel_se = np.sqrt((a_hat ** 2) * (se_b ** 2) + (b_hat ** 2) * (se_a ** 2))
                        sobel_z = indirect / sobel_se if sobel_se else np.nan
                        sobel_p = 2 * (1 - t.cdf(np.abs(sobel_z), df=max(n_b - 3, 1))) if not np.isnan(sobel_z) else np.nan
                    else:
                        sobel_se, sobel_z, sobel_p = np.nan, np.nan, np.nan
                    prop = (indirect / c_hat) if (not np.isnan(indirect) and not np.isnan(c_hat) and c_hat != 0) else np.nan

                    def fmt(v, digits=2):
                        return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{digits}f}"

                    def fmt_p(v):
                        if v is None or (isinstance(v, float) and np.isnan(v)):
                            return "-"
                        return "< .001" if v < 0.001 else f"{v:.3f}"

                    st.session_state.med_completed = True
                    st.session_state.med_summary = {
                        "x": x_label, "m": m_label, "y": dependent_var_label,
                        "a": a_hat, "se_a": se_a, "p_a": p_a,
                        "b": b_hat, "se_b": se_b, "p_b": p_b,
                        "c": c_hat, "se_c": se_c, "p_c": p_c,
                        "cp": cp_hat, "se_cp": se_cp, "p_cp": p_cp,
                        "ind": indirect, "sobel_se": sobel_se, "sobel_z": sobel_z, "sobel_p": sobel_p,
                        "prop": prop, "n": n_b, "n0": n0_b,
                    }
                    st.session_state.med_res_a = (res_a, r2_a, r2a_a, diag_a, n_a, n0_a)
                    st.session_state.med_res_b = (res_b, r2_b, r2a_b, diag_b, n_b, n0_b)
                    st.session_state.med_res_c = (res_c, r2_c, r2a_c, diag_c, n_c, n0_c)

                    st.subheader("Path summary")
                    st.markdown(
                        f"""
| Path | Estimate | SE | *p* |
|---|---:|---:|---:|
| a (X → M) | {fmt(a_hat)} | {fmt(se_a)} | {fmt_p(p_a)} |
| b (M → Y \| X) | {fmt(b_hat)} | {fmt(se_b)} | {fmt_p(p_b)} |
| c (X → Y total) | {fmt(c_hat)} | {fmt(se_c)} | {fmt_p(p_c)} |
| c′ (X → Y direct) | {fmt(cp_hat)} | {fmt(se_cp)} | {fmt_p(p_cp)} |
| a × b (indirect) | {fmt(indirect)} | {fmt(sobel_se)} | {fmt_p(sobel_p)} |
"""
                    )
                    st.caption(
                        f"Sobel z = {fmt(sobel_z, 2)}. "
                        f"Indirect / total = {fmt(prop, 3) if not np.isnan(prop) else '—'}. "
                        f"N (Y model after listwise deletion) = {n_b:,} of {n0_b:,}. "
                        "Sobel is a large-sample test; treat borderline *p* values cautiously. "
                        "Not a causal claim."
                    )
                    cond_rows = []
                    if w_code and not np.isnan(w_sd) and w_sd > 0:
                        a0 = a_hat
                        b0 = b_hat
                        axw = row_xw[1] if row_xw else 0.0
                        se_axw = row_xw[3] if row_xw else np.nan
                        bmw = row_mw[1] if row_mw else 0.0
                        se_bmw = row_mw[3] if row_mw else np.nan
                        if w_raw.nunique(dropna=True) <= 2:
                            levels = [(f"{w_label} = {v:g}", (v - w_mean) if center_ix else v) for v in sorted(w_raw.dropna().unique())]
                        else:
                            levels = [
                                (f"Low {w_label} (−1 SD)", -w_sd if center_ix else w_mean - w_sd),
                                (f"Mean {w_label}", 0.0 if center_ix else w_mean),
                                (f"High {w_label} (+1 SD)", w_sd if center_ix else w_mean + w_sd),
                            ]
                        st.subheader("Conditional indirect effect a(W) × b(W)")
                        st.caption(
                            f"W moderates {which_paths}. "
                            "a(W) = a + a_XW×W and/or b(W) = b + b_MW×W. "
                            "SE uses a Sobel-style delta method at each level (ignores coef covariance)."
                        )
                        md = "| Level of W | a(W) | b(W) | Indirect | SE | *p* |\n|---|---:|---:|---:|---:|---:|\n"
                        for lab, z in levels:
                            az = a0 + (axw if mod_a else 0.0) * z
                            bz = b0 + (bmw if mod_b else 0.0) * z
                            indz = az * bz
                            se_az = np.sqrt(se_a**2 + (z**2)*(se_axw**2)) if (mod_a and not np.isnan(se_a) and not np.isnan(se_axw)) else se_a
                            se_bz = np.sqrt(se_b**2 + (z**2)*(se_bmw**2)) if (mod_b and not np.isnan(se_b) and not np.isnan(se_bmw)) else se_b
                            if not any(np.isnan(v) for v in (az, bz, se_az, se_bz)):
                                se_i = np.sqrt((az**2)*(se_bz**2) + (bz**2)*(se_az**2))
                                zval = indz / se_i if se_i else np.nan
                                pz = 2 * (1 - t.cdf(np.abs(zval), df=max(n_b - 4, 1))) if not np.isnan(zval) else np.nan
                            else:
                                se_i, pz = np.nan, np.nan
                            cond_rows.append((lab, az, bz, indz, se_i, pz))
                            md += f"| {lab} | {fmt(az)} | {fmt(bz)} | {fmt(indz)} | {fmt(se_i)} | {fmt_p(pz)} |\n"
                        st.markdown(md)
                        if row_xw:
                            st.caption(f"X × W on M (moderator of a): B = {fmt(row_xw[1])}, SE = {fmt(row_xw[3])}, p = {fmt_p(row_xw[4])}")
                        if row_mw:
                            st.caption(f"M × W on Y (moderator of b): B = {fmt(row_mw[1])}, SE = {fmt(row_mw[3])}, p = {fmt_p(row_mw[4])}")
                    st.session_state.med_cond = cond_rows
                    st.session_state.med_which = which_paths
                    st.session_state.med_w_name = w_label
                    st.subheader("M ~ X + covariates (path a)")
                    components.html(render_regression_table(m_label, res_a, r2_a, r2a_a, diag_a, n_a, n0_a), height=320, scrolling=True)
                    st.subheader("Y ~ X + M + covariates (paths b and c′)")
                    components.html(render_regression_table(dependent_var_label, res_b, r2_b, r2a_b, diag_b, n_b, n0_b), height=360, scrolling=True)
                    st.subheader("Y ~ X + covariates (path c, total)")
                    components.html(render_regression_table(dependent_var_label, res_c, r2_c, r2a_c, diag_c, n_c, n0_c), height=320, scrolling=True)
                    status_placeholder.empty()
            except Exception as e:
                st.error(f"Error computing mediation: {str(e)}")
                st.session_state.med_completed = False
        elif st.session_state.get("med_completed") and st.session_state.get("med_summary"):
            s = st.session_state.med_summary
            def fmt(v, digits=2):
                return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{digits}f}"
            def fmt_p(v):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return "-"
                return "< .001" if v < 0.001 else f"{v:.3f}"
            st.subheader("Path summary")
            st.markdown(
                f"""
| Path | Estimate | SE | *p* |
|---|---:|---:|---:|
| a (X → M) | {fmt(s['a'])} | {fmt(s['se_a'])} | {fmt_p(s['p_a'])} |
| b (M → Y \| X) | {fmt(s['b'])} | {fmt(s['se_b'])} | {fmt_p(s['p_b'])} |
| c (X → Y total) | {fmt(s['c'])} | {fmt(s['se_c'])} | {fmt_p(s['p_c'])} |
| c′ (X → Y direct) | {fmt(s['cp'])} | {fmt(s['se_cp'])} | {fmt_p(s['p_cp'])} |
| a × b (indirect) | {fmt(s['ind'])} | {fmt(s['sobel_se'])} | {fmt_p(s['sobel_p'])} |
"""
            )
            res_a, r2_a, r2a_a, diag_a, n_a, n0_a = st.session_state.med_res_a
            res_b, r2_b, r2a_b, diag_b, n_b, n0_b = st.session_state.med_res_b
            res_c, r2_c, r2a_c, diag_c, n_c, n0_c = st.session_state.med_res_c
            st.subheader("M ~ X + covariates (path a)")
            components.html(render_regression_table(s["m"], res_a, r2_a, r2a_a, diag_a, n_a, n0_a), height=320, scrolling=True)
            st.subheader("Y ~ X + M + covariates (paths b and c′)")
            components.html(render_regression_table(s["y"], res_b, r2_b, r2a_b, diag_b, n_b, n0_b), height=360, scrolling=True)
            st.subheader("Y ~ X + covariates (path c, total)")
            components.html(render_regression_table(s["y"], res_c, r2_c, r2a_c, diag_c, n_c, n0_c), height=320, scrolling=True)
        else:
            st.write("Choose Y, X and M, then click Run mediation.")