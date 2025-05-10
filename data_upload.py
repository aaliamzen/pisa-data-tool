import streamlit as st
import pandas as pd
import numpy as np
import pyreadstat
import tempfile
import os
import html
import re
from scipy.stats import norm

# Add logo that persists across all pages
try:
    st.logo("assets/logo.png")  # Replace with the path to your logo file, e.g., "assets/logo.png"
except Exception as e:
    st.error(f"Failed to load logo: {e}")

# Streamlit app configuration
st.set_page_config(page_title="PISA Data Exploration Tool", layout="wide")


# Initialize session state for data
if 'df' not in st.session_state:
    st.session_state.df = None
if 'variable_labels' not in st.session_state:
    st.session_state.variable_labels = {}
if 'value_labels' not in st.session_state:
    st.session_state.value_labels = {}
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'visible_columns' not in st.session_state:
    st.session_state.visible_columns = []


# Function to safely convert label to string
def safe_label_to_string(label):
    if label is None:
        return "No description available"
    if isinstance(label, (str, int, float)):
        if isinstance(label, str):
            try:
                label = label.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
            except (UnicodeEncodeError, UnicodeDecodeError):
                label = "Invalid encoding in label"
        return str(label)
    return "No description available"

# Function to apply value labels to DataFrame
def apply_value_labels(df, value_labels):
    df_copy = df.copy()
    for col in df_copy.columns:
        if col in value_labels:
            df_copy[col] = df_copy[col].map(lambda x: value_labels[col].get(x, x) if pd.notnull(x) else x)
    return df_copy

# Function to load PISA data (SPSS only)
def load_pisa_data(uploaded_file, cycle=None):
    # Initialize variables to ensure they are always defined
    df = None
    variable_labels = {}
    value_labels = {}
    
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        try:
            if file_extension == "sav":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".sav") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name
                df, meta = pyreadstat.read_sav(tmp_file_path)
                raw_labels = getattr(meta, 'column_labels', {})
                if isinstance(raw_labels, list):
                    if len(raw_labels) == len(meta.column_names):
                        variable_labels = {
                            col: safe_label_to_string(label)
                            for col, label in zip(meta.column_names, raw_labels)
                        }
                    else:
                        st.warning("Mismatch between column names and labels. Using default labels.")
                        variable_labels = {col: "No description available" for col in df.columns}
                elif isinstance(raw_labels, dict):
                    variable_labels = {
                        col: safe_label_to_string(raw_labels.get(col))
                        for col in df.columns
                    }
                else:
                    st.warning("No valid variable labels found in SPSS metadata. Using default labels.")
                    variable_labels = {col: "No description available" for col in df.columns}
                value_labels = getattr(meta, 'variable_value_labels', {})
                # Check for replicate weights and warn if missing
                replicate_weight_cols = [f"W_FSTURWT{i}" for i in range(1, 81)]
                missing_weights = [col for col in replicate_weight_cols if col not in df.columns]
                if missing_weights:
                    st.warning(f"Missing replicate weights in the dataset: {', '.join(missing_weights)}. Some analyses may use a simpler significance calculation.")
                sample_labels = {k: v for k, v in list(variable_labels.items())[:5]}
                problematic_labels = {col: f"Type: {str(type(label))}, Value: {str(label)[:50]}" 
                                   for col, label in variable_labels.items() if not isinstance(label, str)}
                if problematic_labels:
                    st.warning(f"Problematic variable labels (non-string types): {problematic_labels}")
                st.info(f"Sample variable labels (first 5 columns): {sample_labels}")
                os.unlink(tmp_file_path)
                st.success(f"Loaded PISA SPSS data successfully!")
            else:
                st.error("Unsupported file format. Please upload a .sav file.")
                df = None
                variable_labels = {}
                value_labels = {}
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            df = None
            variable_labels = {}
            value_labels = {}
    return df, variable_labels, value_labels

# Function to render DataFrame as HTML table with tooltips
def dataframe_to_html_with_tooltips(df, variable_labels, value_labels, title="DataFrame", scrollable=False):
    id_columns = ["CNTRYID", "CNTSCHID", "CNTSTUID"]
    categorical_columns = list(value_labels.keys())
    html_content = """
    <style>
    .table-container {
        overflow-x: auto;
        overflow-y: auto;
        max-height: 400px;
        width: 100%;
        scrollbar-width: thin;
    }
    .table-container::-webkit-scrollbar {
        height: 8px;
        width: 8px;
    }
    .table-container::-webkit-scrollbar-thumb {
        background-color: #888;
        border-radius: 4px;
    }
    table {
        width: 100%;
        min-width: {{table_width}}px;
        border-collapse: collapse;
        font-size: 12px;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
        min-width: 100px;
    }
    th {
        background-color: #f2f2f2;
    }
    tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
        padding: 4px;
        font-size: 12px;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 300px;
        background-color: #333;
        color: #fff;
        text-align: left;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 1000;
        top: 100%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
    <div class="table-container">
        <h3>{{title}}</h3>
        <table>
            <tr>
                {{header_row}}
            </tr>
            {{data_rows}}
        </table>
    </div>
    """
    table_width = max(len(df.columns) * 100, 2000)
    header_row = ""
    for col in df.columns:
        label = variable_labels.get(col, "No description available")
        try:
            description = html.escape(str(label))
        except Exception as e:
            st.error(f"Error escaping label for column '{col}': Label={str(label)[:50]}, Type={type(label)}, Error={str(e)}")
            description = "Invalid label"
        header_row += f'<th><div class="tooltip">{col}<span class="tooltiptext">{description}</span></div></th>'
    data_rows = ""
    for _, row in df.iterrows():
        data_rows += "<tr>"
        for col, val in row.items():
            if pd.isna(val):
                formatted_val = "-"
            elif col in id_columns or col in categorical_columns:
                try:
                    if isinstance(val, (int, float)) and float(val).is_integer():
                        formatted_val = str(int(val))
                    else:
                        formatted_val = str(val)
                except (ValueError, TypeError):
                    formatted_val = str(val)
            else:
                formatted_val = str(val)
            data_rows += f"<td>{formatted_val}</td>"
        data_rows += "</tr>"
    html_content = html_content.replace("{{table_width}}", str(table_width))
    html_content = html_content.replace("{{title}}", title)
    html_content = html_content.replace("{{header_row}}", header_row)
    html_content = html_content.replace("{{data_rows}}", data_rows)
    return html_content

# Streamlit UI
st.title("PISA Data Exploration Tool")

# Sidebar for data selection
st.sidebar.header("Data Selection")
uploaded_file = st.sidebar.file_uploader("Upload PISA Data File (.sav)", type=["sav"])
show_value_labels = st.sidebar.checkbox("Show Value Labels", value=False)

# Load data and update session state
df, variable_labels, value_labels = load_pisa_data(uploaded_file)
if df is not None:
    st.session_state.df = df
    st.session_state.variable_labels = variable_labels
    st.session_state.value_labels = value_labels
    st.session_state.data_loaded = True
else:
    st.session_state.data_loaded = False

# Main content
st.header("Data Summary")
if df is None:
    st.info("Please upload a PISA .sav file to view the data preview.")
else:
    # Country selection
    st.sidebar.header("Data Filtering")
    country_col = "CNTRYID" if "CNTRYID" in df.columns else "CNT"
    if country_col in df.columns:
        country_values = sorted(df[country_col].unique())
        if country_col in value_labels:
            country_labels = [value_labels[country_col].get(val, str(val)) for val in country_values]
            label_to_value = {value_labels[country_col].get(val, str(val)): val for val in country_values}
        else:
            country_labels = [str(val) for val in country_values]
            label_to_value = {str(val): val for val in country_values}
        selected_labels = st.sidebar.multiselect("Select Countries", country_labels, default=country_labels[:2] if country_labels else [])
        selected_countries = [label_to_value[label] for label in selected_labels if label in label_to_value]
        if selected_countries:
            df = df[df[country_col].isin(selected_countries)]
            st.session_state.df = df  # Update session state with filtered data
    else:
        st.sidebar.warning("No country column (CNTRYID or CNT) found in the dataset.")

    if df.empty:
        st.warning("No data available after filtering by selected countries.")
    else:
        # Define weight columns (including replicate weights, which are hidden from display)
        weight_columns = ['W_FSTUWT'] + [f"W_FSTURWT{i}" for i in range(1, 81)] + [col for col in df.columns if 'W_FSCHWT' in col]
        excluded_variables = [
            'CYC', 'NatCen', 'STRATUM', 'SUBNATIO', 'REGION', 'OECD', 'ADMINMODE',
            'LANGTEST_QQQ', 'LANGTEST_COG', 'LANGTEST_PAQ', 'Option_CT', 'Option_FL',
            'Option_ICTQ', 'Option_WBQ', 'Option_PQ', 'Option_TQ', 'Option_UH', 'BOOKID',
            'COBN_S', 'COBN_M', 'COBN_F', 'OCOD1', 'OCOD2', 'OCOD3',
            'ST001D01T', 'ST003D02T', 'ST003D03T',
            'EFFORT1', 'EFFORT2', 'PROGN', 'ISCEDP', 'SENWT', 'VER_DAT', 'test',
            'GRADE', 'UNIT', 'WVARSTRR',
            'PAREDINT', 'HISEI', 'HOMEPOS', 'BMMJ1', 'BFMJ2',
            'SCHOOLID', 'STUID', 'CNT', 'CNTRYID', 'CNTSCHID', 'CNTSTUID'
        ]
        item_pattern = re.compile(r'^(ST|FL|IC|WB|PA)\w{8}$', re.IGNORECASE)
        pv_pattern = re.compile(r'^PV([1-9]|10)[A-Z]+(\d*)$', re.IGNORECASE)  # Match any PV variable
        display_columns = [
            col for col in df.columns 
            if col not in weight_columns 
            and col not in excluded_variables
            and not item_pattern.match(col) 
            and not pv_pattern.match(col)  # Exclude all PV variables
            and not df[col].isna().all()
        ]
        st.session_state.visible_columns = display_columns  # Store visible columns for downstream scripts
        if not display_columns:
            st.warning("No non-missing data available for display after filtering.")
        else:
            st.info(f"Visible columns in preview table: {', '.join(display_columns)}")
            display_df = df[display_columns]
            if show_value_labels:
                display_df = apply_value_labels(display_df, value_labels)
                if not any(col in value_labels for col in display_columns):
                    st.warning("No value labels available for visible columns in the dataset.")
            slipped_display_items = [
                col for col in display_columns 
                if item_pattern.match(col) 
                or col in excluded_variables 
                or pv_pattern.match(col)
            ]
            if slipped_display_items:
                st.warning(f"Excluded items or non-PV1 plausible values detected in display: {slipped_display_items}")
            st.write(f"Dataset Size: {df.shape[0]} rows, {display_df.shape[1]} columns (weights, questionnaire items, administrative variables, ESCS components, and fully missing columns hidden)")
            preview_html = dataframe_to_html_with_tooltips(display_df.head(10), variable_labels, value_labels, title="Data Preview", scrollable=True)
            st.markdown(preview_html, unsafe_allow_html=True)

# Instructions section
st.header("Instructions")
st.markdown("""
- **Upload Data**: Use the sidebar to upload a PISA .sav file.
- **Select Countries**: Choose one or more countries to filter the data in the sidebar.
- **Navigate to Analysis**: Use the sidebar navigation to select an analysis type (Bivariate Correlation, Correlation Matrix, or Linear Regression) and perform your analysis.
""")