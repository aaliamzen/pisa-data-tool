import streamlit as st
import pandas as pd
import numpy as np
import re
import streamlit.components.v1 as components
from io import BytesIO
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# Streamlit app configuration
st.set_page_config(page_title="Descriptive Statistics - PISA Data Exploration Tool", layout="wide")

# Function to classify variables as continuous or categorical
def classify_variable(series):
    # Treat numeric variables (float64, int64) with more than 10 unique values as continuous
    if series.dtype in ['float64', 'int64'] and series.nunique() > 10:
        return 'continuous'
    # Treat other variables (including numerics with few unique values) as categorical
    else:
        return 'categorical'

# Function to compute weighted mean
def weighted_mean(values, weights):
    mask = ~(np.isnan(values) | np.isnan(weights))
    values = values[mask]
    weights = weights[mask]
    if len(values) == 0 or np.sum(weights) == 0:
        return np.nan
    return np.sum(values * weights) / np.sum(weights)

# Function to compute weighted standard deviation
def weighted_std(values, weights):
    mask = ~(np.isnan(values) | np.isnan(weights))
    values = values[mask]
    weights = weights[mask]
    if len(values) < 2 or np.sum(weights) == 0:
        return np.nan
    mean = weighted_mean(values, weights)
    variance = np.sum(weights * (values - mean)**2) / np.sum(weights)
    return np.sqrt(variance)

# Function to compute weighted quantiles (e.g., median, min, max)
def weighted_quantile(values, weights, quantile=0.5):
    mask = ~(np.isnan(values) | np.isnan(weights))
    values = values[mask]
    weights = weights[mask]
    if len(values) == 0 or np.sum(weights) == 0:
        return np.nan
    sorted_indices = np.argsort(values)
    values = values[sorted_indices]
    weights = weights[sorted_indices]
    cumsum = np.cumsum(weights)
    target = quantile * cumsum[-1]
    for i in range(len(cumsum)):
        if cumsum[i] >= target:
            return values[i]
    return values[-1]

# Function to compute weighted skewness
def weighted_skewness(values, weights):
    mask = ~(np.isnan(values) | np.isnan(weights))
    values = values[mask]
    weights = weights[mask]
    if len(values) < 2 or np.sum(weights) == 0:
        return np.nan
    mean = weighted_mean(values, weights)
    m3 = np.sum(weights * (values - mean)**3) / np.sum(weights)
    m2 = np.sum(weights * (values - mean)**2) / np.sum(weights)
    if m2 == 0:
        return np.nan
    return m3 / (m2 ** 1.5)

# Function to compute weighted kurtosis (excess kurtosis)
def weighted_kurtosis(values, weights):
    mask = ~(np.isnan(values) | np.isnan(weights))
    values = values[mask]
    weights = weights[mask]
    if len(values) < 2 or np.sum(weights) == 0:
        return np.nan
    mean = weighted_mean(values, weights)
    m4 = np.sum(weights * (values - mean)**4) / np.sum(weights)
    m2 = np.sum(weights * (values - mean)**2) / np.sum(weights)
    if m2 == 0:
        return np.nan
    return (m4 / (m2 ** 2)) - 3

# Function to compute weighted sample size (sum of weights for non-NA values)
def weighted_sample_size(values, weights):
    mask = ~(np.isnan(values) | np.isnan(weights))
    weights = weights[mask]
    return np.sum(weights)

# Function to compute descriptive statistics with weights
def compute_descriptive_stats(df, selected_vars, variable_labels, value_labels, weights):
    stats = []
    for var_label in selected_vars:
        # Get the variable code from the label
        var_code = next((code for code, label in variable_labels.items() if label == var_label), var_label)
        if var_code not in df.columns:
            continue
        
        series = df[var_code]
        var_type = classify_variable(series)
        missing_count = series.isna().sum()
        missing_percent = (missing_count / len(series)) * 100
        
        if var_type == 'continuous':
            mean = weighted_mean(series.values, weights)
            sd = weighted_std(series.values, weights)
            min_val = weighted_quantile(series.values, weights, quantile=0.0)  # Weighted min
            max_val = weighted_quantile(series.values, weights, quantile=1.0)  # Weighted max
            skewness = weighted_skewness(series.values, weights)
            kurtosis = weighted_kurtosis(series.values, weights)
            n = weighted_sample_size(series.values, weights)
            stats.append({
                'Variable': var_label,
                'Type': 'Continuous',
                'N': n,
                'Mean': mean,
                'SD': sd,
                'Min': min_val,
                'Max': max_val,
                'Skewness': skewness,
                'Kurtosis': kurtosis,
                'Missing (%)': missing_percent,
                'Median': '-',
                'Mode': '-',
                'Category': '-',
                'Count': '-',
                'Percent': '-'
            })
        else:
            # Treat as categorical
            value_counts = series.value_counts(dropna=False)
            total_count = len(series)
            total_weight = np.sum(weights[~series.isna()])
            median_val = weighted_quantile(series.values, weights) if series.dtype in ['float64', 'int64'] else 'N/A'
            # Compute mode as the category with the highest weighted frequency
            categories = value_counts.index
            weighted_counts = {}
            for cat in categories:
                if pd.isna(cat):
                    weighted_counts[cat] = 0
                else:
                    mask = series == cat
                    weighted_counts[cat] = np.sum(weights[mask])
            mode_val = max(weighted_counts.items(), key=lambda x: x[1])[0] if weighted_counts else 'N/A'
            mode_label = value_labels.get(var_code, {}).get(mode_val, mode_val)
            
            # For categorical variables, create a single row with concatenated categories
            categories_str = []
            counts_str = []
            percents_str = []
            for category, count in value_counts.items():
                if pd.isna(category):
                    continue
                # Use value labels if available
                category_label = value_labels.get(var_code, {}).get(category, category)
                mask = series == category
                weighted_count = np.sum(weights[mask])
                percent = (weighted_count / total_weight) * 100 if total_weight > 0 else 0
                categories_str.append(str(category_label))
                counts_str.append(str(int(weighted_count)))
                percents_str.append(f"{percent:.2f}%")
            
            stats.append({
                'Variable': var_label,
                'Type': 'Categorical',
                'N': '-',
                'Mean': '-',
                'SD': '-',
                'Min': '-',
                'Max': '-',
                'Skewness': '-',
                'Kurtosis': '-',
                'Median': median_val,
                'Mode': mode_label,
                'Category': "; ".join(categories_str),
                'Count': "; ".join(counts_str),
                'Percent': "; ".join(percents_str),
                'Missing (%)': missing_percent
            })
    return stats

# Function to render APA-style table as HTML
def render_descriptive_table(stats):
    html_content = """
    <style>
    .desc-table-container {
        display: inline-block;
        overflow-x: auto;
        scrollbar-width: thin;
        min-width: 0;
        margin: 20px 0;
    }
    .desc-table-container::-webkit-scrollbar {
        height: 8px;
    }
    .desc-table-container::-webkit-scrollbar-thumb {
        background-color: #888;
        border-radius: 4px;
    }
    .desc-table {
        table-layout: fixed;
        border-collapse: collapse;
        font-size: 14px;
        margin: 0;
    }
    .desc-table th, .desc-table td {
        border: none;
        padding: 8px;
        box-sizing: border-box;
        text-align: center;
    }
    .desc-table th:first-child, .desc-table td:first-child {
        width: 200px !important;
        text-align: left;
        white-space: normal;
        overflow-wrap: break-word;
    }
    .desc-table th:not(:first-child), .desc-table td:not(:first-child) {
        width: 100px !important;
    }
    .desc-table tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    .desc-table-title {
        font-size: 16px;
        font-weight: bold;
        text-align: left;
        margin-bottom: 5px;
    }
    .desc-table-subtitle {
        font-size: 16px;
        font-style: italic;
        text-align: left;
        margin-bottom: 10px;
    }
    .desc-table-header {
        border-top: 1px solid #000;
        border-bottom: 1px solid #000;
    }
    .desc-table-last-row {
        border-bottom: 1px solid #000;
    }
    </style>
    <div class="desc-table-container">
        <div class="desc-table-title">Table 1</div>
        <div class="desc-table-subtitle">Weighted Descriptive Statistics of Selected Variables (Using Final Student Weights)</div>
        <table class="desc-table">
            <tr class="desc-table-header">
                <th>Variable</th>
                <th>Type</th>
                <th>N</th>
                <th>Mean</th>
                <th>SD</th>
                <th>Min</th>
                <th>Max</th>
                <th>Skewness</th>
                <th>Kurtosis</th>
                <th>Median</th>
                <th>Mode</th>
                <th>Category</th>
                <th>Count</th>
                <th>Percent</th>
                <th>Missing (%)</th>
            </tr>
            {{data_rows}}
        </table>
    </div>
    """
    data_rows = ""
    for idx, stat in enumerate(stats):
        row_class = "desc-table-last-row" if idx == len(stats) - 1 else ""
        # Use get() with a default value to handle missing keys
        median = stat.get('Median', '-')
        mode = stat.get('Mode', '-')
        category = stat.get('Category', '-')
        count = stat.get('Count', '-')
        percent = stat.get('Percent', '-')
        # Format continuous variable statistics to 2 DP
        mean = f"{stat['Mean']:.2f}" if isinstance(stat['Mean'], (int, float)) and not np.isnan(stat['Mean']) else stat['Mean']
        sd = f"{stat['SD']:.2f}" if isinstance(stat['SD'], (int, float)) and not np.isnan(stat['SD']) else stat['SD']
        min_val = f"{stat['Min']:.2f}" if isinstance(stat['Min'], (int, float)) and not np.isnan(stat['Min']) else stat['Min']
        max_val = f"{stat['Max']:.2f}" if isinstance(stat['Max'], (int, float)) and not np.isnan(stat['Max']) else stat['Max']
        skewness = f"{stat['Skewness']:.2f}" if isinstance(stat['Skewness'], (int, float)) and not np.isnan(stat['Skewness']) else stat['Skewness']
        kurtosis = f"{stat['Kurtosis']:.2f}" if isinstance(stat['Kurtosis'], (int, float)) and not np.isnan(stat['Kurtosis']) else stat['Kurtosis']
        n = f"{stat['N']:.2f}" if isinstance(stat['N'], (int, float)) and not np.isnan(stat['N']) else stat['N']
        row = f"""
        <tr class="{row_class}">
            <th>{stat['Variable']}</th>
            <td>{stat['Type']}</td>
            <td>{n}</td>
            <td>{mean}</td>
            <td>{sd}</td>
            <td>{min_val}</td>
            <td>{max_val}</td>
            <td>{skewness}</td>
            <td>{kurtosis}</td>
            <td>{median}</td>
            <td>{mode}</td>
            <td>{category}</td>
            <td>{count}</td>
            <td>{percent}</td>
            <td>{stat['Missing (%)']:.2f}%</td>
        </tr>
        """
        data_rows += row
    
    full_html = html_content.replace("{{data_rows}}", data_rows)
    return full_html

# Function to set cell borders in Word document
def set_cell_border(cell, top=False, bottom=False, left=False, right=False):
    tc = cell._element
    tcPr = tc.get_or_add_tcPr()
    
    # Remove existing borders
    for border in ['top', 'bottom', 'left', 'right']:
        existing_border = tcPr.find(qn(f'w:{border}'))
        if existing_border is not None:
            tcPr.remove(existing_border)
    
    # Set new borders where specified
    if top:
        border = OxmlElement('w:top')
        border.set(qn('w:val'), 'single')
        border.set(qn('w:sz'), '4')  # Border width in eighths of a point
        tcPr.append(border)
    if bottom:
        border = OxmlElement('w:bottom')
        border.set(qn('w:val'), 'single')
        border.set(qn('w:sz'), '4')
        tcPr.append(border)
    if left:
        border = OxmlElement('w:left')
        border.set(qn('w:val'), 'single')
        border.set(qn('w:sz'), '4')
        tcPr.append(border)
    if right:
        border = OxmlElement('w:right')
        border.set(qn('w:val'), 'single')
        border.set(qn('w:sz'), '4')
        tcPr.append(border)

# Function to create a Word document with the APA-style table
def create_word_table(stats):
    doc = Document()
    
    # Set document to landscape orientation
    section = doc.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    # Adjust page dimensions for landscape (swap width and height)
    section.page_width = Inches(11)
    section.page_height = Inches(8.5)
    
    # Add title
    title = doc.add_paragraph("Table 1")
    title.runs[0].bold = True
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    title.runs[0].font.name = 'Times New Roman'
    title.runs[0].font.size = Pt(10)
    
    # Add subtitle
    subtitle = doc.add_paragraph("Weighted Descriptive Statistics of Selected Variables (Using Final Student Weights)")
    subtitle.runs[0].italic = True
    subtitle.alignment = WD_ALIGN_PARAGRAPH.LEFT
    subtitle.runs[0].font.name = 'Times New Roman'
    subtitle.runs[0].font.size = Pt(10)
    
    # Create table
    table = doc.add_table(rows=1 + len(stats), cols=15)
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table.style = 'Normal Table'  # Use a style with no borders
    
    # Set column widths
    for column in table.columns:
        for cell in column.cells:
            cell.width = Inches(0.7)  # Adjusted width for landscape orientation
    
    # Add header row
    headers = [
        "Variable", "Type", "N", "Mean", "SD", "Min", "Max", "Skewness", "Kurtosis",
        "Median", "Mode", "Category", "Count", "Percent", "Missing (%)"
    ]
    for idx, header in enumerate(headers):
        cell = table.rows[0].cells[idx]
        cell.text = header
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cell.paragraphs[0].runs[0].font.name = 'Times New Roman'
        cell.paragraphs[0].runs[0].font.size = Pt(10)
        cell.paragraphs[0].runs[0].bold = True
        # Add top and bottom borders to header row
        set_cell_border(cell, top=True, bottom=True)
    
    # Add data rows
    for idx, stat in enumerate(stats):
        row = table.rows[idx + 1]
        row.cells[0].text = stat['Variable']
        row.cells[1].text = stat['Type']
        row.cells[2].text = f"{stat['N']:.2f}" if isinstance(stat['N'], (int, float)) and not np.isnan(stat['N']) else stat['N']
        row.cells[3].text = f"{stat['Mean']:.2f}" if isinstance(stat['Mean'], (int, float)) and not np.isnan(stat['Mean']) else stat['Mean']
        row.cells[4].text = f"{stat['SD']:.2f}" if isinstance(stat['SD'], (int, float)) and not np.isnan(stat['SD']) else stat['SD']
        row.cells[5].text = f"{stat['Min']:.2f}" if isinstance(stat['Min'], (int, float)) and not np.isnan(stat['Min']) else stat['Min']
        row.cells[6].text = f"{stat['Max']:.2f}" if isinstance(stat['Max'], (int, float)) and not np.isnan(stat['Max']) else stat['Max']
        row.cells[7].text = f"{stat['Skewness']:.2f}" if isinstance(stat['Skewness'], (int, float)) and not np.isnan(stat['Skewness']) else stat['Skewness']
        row.cells[8].text = f"{stat['Kurtosis']:.2f}" if isinstance(stat['Kurtosis'], (int, float)) and not np.isnan(stat['Kurtosis']) else stat['Kurtosis']
        row.cells[9].text = str(stat.get('Median', '-'))
        row.cells[10].text = str(stat.get('Mode', '-'))
        row.cells[11].text = str(stat.get('Category', '-'))
        row.cells[12].text = str(stat.get('Count', '-'))
        row.cells[13].text = str(stat.get('Percent', '-'))
        row.cells[14].text = f"{stat['Missing (%)']:.2f}%"
        for cell in row.cells:
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cell.paragraphs[0].runs[0].font.name = 'Times New Roman'
            cell.paragraphs[0].runs[0].font.size = Pt(10)
            # Add bottom border to the last row
            if idx == len(stats) - 1:
                set_cell_border(cell, bottom=True)
    
    # Save document to a BytesIO buffer
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

# Access data from session state
df = st.session_state.get('df', None)
variable_labels = st.session_state.get('variable_labels', {})
value_labels = st.session_state.get('value_labels', {})
visible_columns = st.session_state.get('visible_columns', [])

# Initialize session state for this page
# Reset session state to clear old results
if 'descriptive_stats_results' not in st.session_state:
    st.session_state.descriptive_stats_results = None
if 'descriptive_stats_completed' not in st.session_state:
    st.session_state.descriptive_stats_completed = False

# Streamlit UI
st.title("Descriptive Statistics Analysis")

if df is None or df.empty:
    st.warning("No data available. Please upload a dataset on the main page.")
else:
    # Detect all variables for selection (both numeric and categorical)
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
    selectable_vars_codes = [
        col for col in df.columns 
        if col not in weight_columns 
        and col not in excluded_variables
        and not item_pattern.match(col) 
        and not df[col].isna().all()
        and col in visible_columns
    ]
    
    # Convert variable codes to labels for display
    selectable_vars = []
    for code in selectable_vars_codes:
        label = variable_labels.get(code, code)
        selectable_vars.append(label)
    
    if len(selectable_vars) < 1:
        st.warning("No variables available for descriptive statistics. Please ensure the dataset contains at least one valid variable.")
    else:
        # Select Variables
        st.write("Select variables for descriptive statistics:")
        default_vars = []
        if "Mathematics score" in selectable_vars:
            default_vars.append("Mathematics score")
        escs_label = variable_labels.get("ESCS", "ESCS")
        if escs_label in selectable_vars:
            default_vars.append(escs_label)
        elif len(selectable_vars) > 1:
            default_vars.append(selectable_vars[1 if "Mathematics score" in selectable_vars else 0])
        
        selected_vars = st.multiselect(
            "Variables",
            selectable_vars,
            default=default_vars,
            key="desc_vars"
        )
        
        run_analysis = st.button("Run Analysis", key="run_descriptive_stats")
        
        if run_analysis and selected_vars:
            try:
                if 'W_FSTUWT' not in df.columns:
                    st.error("Final student weight (W_FSTUWT) not found in the dataset.")
                else:
                    weights = df['W_FSTUWT'].values
                    # Compute descriptive statistics
                    stats = compute_descriptive_stats(df, selected_vars, variable_labels, value_labels, weights)
                    
                    # Store results in session state
                    st.session_state.descriptive_stats_results = stats
                    st.session_state.descriptive_stats_completed = True
                    
                    # Render the APA-style table
                    table_html = render_descriptive_table(stats)
                    components.html(table_html, height=400, scrolling=True)
                    st.write("Descriptive statistics analysis completed.")
                    
                    # Provide download button for Word document
                    doc_buffer = create_word_table(stats)
                    st.download_button(
                        label="Download Table as Word Document",
                        data=doc_buffer,
                        file_name="Descriptive_Statistics_Table.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
            except Exception as e:
                st.error(f"Error computing descriptive statistics: {str(e)}")
                st.session_state.descriptive_stats_completed = False
                st.session_state.descriptive_stats_results = None
        elif st.session_state.descriptive_stats_results and st.session_state.descriptive_stats_completed:
            if selected_vars:
                table_html = render_descriptive_table(st.session_state.descriptive_stats_results)
                components.html(table_html, height=400, scrolling=True)
                st.write("Descriptive statistics analysis completed.")
                
                # Provide download button for Word document
                doc_buffer = create_word_table(st.session_state.descriptive_stats_results)
                st.download_button(
                    label="Download Table as Word Document",
                    data=doc_buffer,
                    file_name="Descriptive_Statistics_Table.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            else:
                st.write("Please select at least one variable to compute descriptive statistics.")
        else:
            st.write("Please select at least one variable and click 'Run Analysis' to compute descriptive statistics.")

# Instructions section
st.header("Instructions")
st.markdown("""
- **Select Variables**: Choose one or more variables from the dropdown menu. The tool will classify each variable as continuous or categorical based on its data type and number of unique values.
- **Run Analysis**: Click "Run Analysis" to compute weighted descriptive statistics using student weights (W_FSTUWT). Continuous variables will show weighted mean, weighted SD, range, skewness, kurtosis, and sample size. Categorical variables will show weighted median, mode, weighted counts, and percentages per category.
- **View Results**: Results are displayed in an APA-style table using variable labels. For categorical variables, category names are displayed using value labels. You can download the table as a Word document using the download button.
- **Navigate**: Use the sidebar to switch between different analysis types or return to the main page to upload a new dataset.
""")