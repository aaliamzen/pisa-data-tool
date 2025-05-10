#Imports and Data Checks
import streamlit as st  
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns
import textwrap
from io import BytesIO
from docx import Document  
from docx.shared import Pt, Inches  
from docx.enum.section import WD_ORIENT  
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT

# Add logo that persists across all pages
try:
    st.logo("assets/logo.png")  # Replace with the path to your logo file, e.g., "assets/logo.png"
except Exception as e:
    st.error(f"Failed to load logo: {e}")
  
st.set_page_config(page_title="PISA Data Exploration Tool - Descriptive Statistics", layout="wide")  
  
if 'df' not in st.session_state or st.session_state.df is None:  
    st.error("Please upload your data first in the Data Upload page!")  
    st.stop()  
  
df = st.session_state.df  
variable_labels = st.session_state.variable_labels  
value_labels = st.session_state.value_labels  
visible_columns = st.session_state.visible_columns  
  
if 'W_FSTUWT' not in df.columns:  
    st.error("The final student weight (W_FSTUWT) is not present in your data. Weighted statistics cannot be computed.")  
    st.stop() 
    

#Variable Label Mapping and User Selection

var_code_to_label = {code: label for code, label in variable_labels.items() if code in visible_columns}  
var_label_to_code = {label: code for code, label in var_code_to_label.items()}  
  
st.title("Descriptive Statistics")  
st.markdown("""  
All statistics and visualizations below use the final student weight (W_FSTUWT) for accurate population estimates.  
Select the variables you want to analyze below.  
""")  
  
selected_labels = st.multiselect(    
    "Select Variables for Analysis",    
    options=[var_code_to_label[code] for code in visible_columns],    
    default=[],  # No default selection  
    help="Choose the variables you want to analyze"    
)    
selected_variables = [var_label_to_code[label] for label in selected_labels]  

def is_categorical(series):  
    """Detect if a variable is categorical based on dtype and unique values"""  
    # If it's already categorical dtype or object (usually strings)  
    if pd.api.types.is_categorical_dtype(series) or pd.api.types.is_object_dtype(series):  
        return True  
    # If numeric, check if it has a small number of unique values (e.g., Likert scales)  
    if pd.api.types.is_numeric_dtype(series):  
        n_unique = series.nunique()  
        return n_unique <= 10  # Adjust this threshold as needed  
    return False  

#Weighted Descriptive Functions

# Statistical functions for PISA data analysis

import numpy as np
import pandas as pd

def weighted_mean(x, w):
    """Calculate weighted mean."""
    return np.sum(x * w) / np.sum(w)

def weighted_var(x, w):
    """Calculate weighted variance."""
    return np.sum(w * (x - weighted_mean(x, w))**2) / np.sum(w)

def weighted_std(x, w):
    """Calculate weighted standard deviation."""
    return np.sqrt(weighted_var(x, w))

def weighted_skew(x, w):
    """Calculate weighted skewness."""
    mean = weighted_mean(x, w)
    std = weighted_std(x, w)
    return np.sum(w * ((x - mean)/std)**3) / np.sum(w)

def weighted_kurtosis(x, w):
    """Calculate weighted kurtosis."""
    mean = weighted_mean(x, w)
    std = weighted_std(x, w)
    return np.sum(w * ((x - mean)/std)**4) / np.sum(w)

def is_categorical(series):
    """
    Determine if a series is categorical based on number of unique values
    and dtype.
    """
    n_unique = len(series.unique())
    return (
        pd.api.types.is_categorical_dtype(series) or 
        pd.api.types.is_object_dtype(series) or 
        (pd.api.types.is_numeric_dtype(series) and n_unique <= 10)
    )

def calculate_weighted_descriptives(df, variables, variable_labels, weights, value_labels):
    """
    Calculate weighted descriptive statistics for selected variables.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The data frame containing the variables
    variables : list
        List of variable names to analyze
    variable_labels : dict
        Dictionary mapping variable names to their labels
    weights : pandas Series
        The weights to use for calculations
    value_labels : dict
        Dictionary mapping categorical values to their labels
        
    Returns:
    --------
    pandas DataFrame
        DataFrame containing descriptive statistics
    """
    results = []
    
    for var in variables:
        if is_categorical(df[var]):
            # Categorical statistics
            mask = df[var].notna() & weights.notna()
            cats = df.loc[mask, var]
            w = weights.loc[mask]
            
            # Handle empty or all-missing case
            if len(cats) == 0 or w.sum() == 0:
                results.append({
                    'Variable': variable_labels.get(var, var),
                    'N': 0,
                    'Mode': "-",
                    'Most Common %': "-"
                })
                continue
            
            # Calculate weighted frequencies
            freq = cats.groupby(cats).apply(lambda x: w.loc[x.index].sum())
            total = freq.sum()
            percentages = (freq / total * 100)
            
            # Find weighted mode and convert to label if available
            mode_cat = freq.index[freq.argmax()]
            if var in value_labels:
                mode_label = value_labels[var].get(mode_cat, mode_cat)
            else:
                mode_label = str(mode_cat)
            
            results.append({
                'Variable': variable_labels.get(var, var),
                'N': int(mask.sum()) if mask.sum() > 0 else "-",
                'Mode': mode_label if mask.sum() > 0 else "-",
                'Most Common %': f"{percentages.max():.1f}%" if mask.sum() > 0 else "-"
            })
        else:
            # Numeric statistics
            mask = df[var].notna() & weights.notna()
            x = df.loc[mask, var]
            w = weights.loc[mask]
            
            # Handle empty or all-missing case
            if len(x) == 0 or w.sum() == 0:
                results.append({
                    'Variable': variable_labels.get(var, var),
                    'N': 0,
                    'Mean': "-",
                    'SD': "-",
                    'Skewness': "-",
                    'Kurtosis': "-"
                })
                continue
            
            results.append({
                'Variable': variable_labels.get(var, var),
                'N': int(mask.sum()) if mask.sum() > 0 else "-",
                'Mean': weighted_mean(x, w) if mask.sum() > 0 else "-",
                'SD': weighted_std(x, w) if mask.sum() > 0 else "-",
                'Skewness': weighted_skew(x, w) if mask.sum() > 0 else "-",
                'Kurtosis': weighted_kurtosis(x, w) if mask.sum() > 0 else "-"
            })
    
    return pd.DataFrame(results)
    

 
#APA Table HTML Generator



def create_apa_table_html(df, title="Descriptive Statistics for Selected Variables"):
    """
    Create an APA-styled HTML table from the descriptive statistics DataFrame.
    Dynamically adapts to the columns present in the DataFrame.
    """
    # CSS styling for APA format
    css = """
    <style>
        .apa-table {
            font-family: "Times New Roman", Times, serif;
            font-size: 12pt;
            border-collapse: collapse;
            margin: 0;
            width: 100%;
            border: none !important;
            background-color: transparent !important;
        }
        .apa-table caption {
            font-style: italic;
            text-align: left;
            margin-bottom: 10px;
        }
        .apa-table th {
            border: none !important;
            border-bottom: 1px solid black !important;
            border-top: 1px solid black !important;
            font-weight: normal;
            padding: 8px;
            text-align: center;
            background: none !important;
            background-color: transparent !important;
        }
        .apa-table td {
            border: none !important;
            padding: 8px;
            background: none !important;
            background-color: transparent !important;
        }
        .apa-table td.left {
            text-align: left;
        }
        .apa-table td.numeric {
            text-align: center;
        }
        .apa-table tr {
            border: none !important;
            background: none !important;
            background-color: transparent !important;
        }
        .apa-table tbody tr {
            border: none !important;
            background: none !important;
            background-color: transparent !important;
        }
        .apa-table tr:last-child td {
            border: none !important;
            border-bottom: 1px solid black !important;
        }
        .table-number {
            font-family: "Times New Roman", Times, serif;
            font-weight: bold;
            text-align: left;
            margin-bottom: 0;
            padding-bottom: 8px;
        }
        .table-note {
            font-family: "Times New Roman", Times, serif;
            font-size: 12pt;
            text-align: left;
            margin-top: 10px;
        }
        .table-note em {
            font-family: "Times New Roman", Times, serif;
            font-style: italic;
        }
    </style>
    """

    # Start building the HTML table
    html = css
    html += '<div class="table-number">Table 1</div>'
    html += '<table class="apa-table">'
    html += f'<caption>{title}</caption>'
    html += '<thead><tr>'

    # Dynamically create headers
    for col in df.columns:
        # Italicize statistical notation if present
        if col in ['N', 'M', 'SD', 'Mean', 'Mode', 'Median']:
            html += f'<th><em>{col}</em></th>'
        else:
            html += f'<th>{col}</th>'
    html += '</tr></thead><tbody>'

    # Add each row of data
    for _, row in df.iterrows():
        html += "<tr>"
        for idx, col in enumerate(df.columns):
            val = row[col]
            # Handle missing values and formatting
            if pd.isna(val) or val == "nan" or val == "":
                cell = "-"
            else:
                try:
                    if isinstance(val, (int, float)) and col != 'N':
                        cell = f"{val:.2f}"
                    else:
                        cell = str(val)
                except Exception:
                    cell = str(val)
            # Left-justify the first column (Variable), center the rest
            if idx == 0:
                html += f"<td class='left'>{cell}</td>"
            else:
                html += f"<td class='numeric'>{cell}</td>"
        html += "</tr>"

    html += """
        </tbody>
    </table>
    <div class="table-note">
        <em>Note.</em> All statistics are weighted using final student weights (W_FSTUWT).
    </div>
    """

    return html


print("APA Table HTML Generator updated: bolded title, top border above headers, bottom border below last row. Saved as 'apa_table_generator_bolded.txt'")
 
  
### 6. Display the Table and Visualizations  
  
##Now, we calculate the weighted descriptives, format the numbers, and display the APA table and plots.  
 
if selected_variables:  
    weights = df["W_FSTUWT"]  
    descriptives_df = calculate_weighted_descriptives(df, selected_variables, variable_labels, weights, value_labels)  
    # Format numeric columns  
    numeric_cols = ["Mean", "SD", "Skewness", "Kurtosis"]  
    for col in numeric_cols:  
        if col in descriptives_df.columns:  
            descriptives_df[col] = descriptives_df[col].apply(lambda x: f"{float(x):.2f}" if x != "" else "")  
    # Show the APA table  
    table_html = create_apa_table_html(descriptives_df)  
    st.markdown(table_html, unsafe_allow_html=True)  

    # Add header for visualizations section
    # Add space and visualization header  
    st.markdown("<br><br>", unsafe_allow_html=True)  # Adds space without duplicating the table       
    st.subheader("Visualisations") 
 
# Show weighted plots for each variable    
    for var, label in zip(selected_variables, selected_labels):  
        #st.markdown(f"#### {label}")  # Using smaller header as discussed  
          
        if not is_categorical(df[var]):  
            # Your existing numeric visualization code  
            mask = df[var].notna() & df["W_FSTUWT"].notna()  
            x = df.loc[mask, var]  
            w = df.loc[mask, "W_FSTUWT"]  
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))  
            ax1.hist(x, bins=30, weights=w, color="skyblue", edgecolor="black")  
            ax1.set_title("Weighted Histogram")  
            ax1.set_xlabel(label)  
            ax1.set_ylabel("Weighted Count")
              
            ax1.legend()  
            plt.tight_layout()  
            plt.show() 
        
            sns.boxplot(y=x, ax=ax2, width=0.5)  
            ax2.set_title("Box Plot")  
            ax2.set_ylabel(label)  
            plt.tight_layout()  
            st.pyplot(fig)  
        else:  
            # Enhanced categorical visualization  
            mask = df[var].notna() & df["W_FSTUWT"].notna()  
            cats = df.loc[mask, var]  
            w = df.loc[mask, "W_FSTUWT"]  

            # Calculate weighted frequencies and percentages  
            freq = cats.groupby(cats).apply(lambda x: w.loc[x.index].sum())  
            total = freq.sum()  
            percentages = (freq / total * 100)  

            # Sort by frequency (you might want to skip this for ordinal variables)  
            #freq = freq.sort_values(ascending=True)  # Changed to True for better visualization  
            #percentages = percentages[freq.index]  

            # Calculate the number of categories  
            num_categories = len(freq)  

            # Dynamically calculate the plot height based on the number of categories  
            base_height = 2.5  # Height for 2 categories  
            base_num_categories = 2  
            height_per_category = 0.5  # Additional height per category  
            height = max(1.0, base_height + (num_categories - base_num_categories) * height_per_category)  # Ensure minimum height of 1  

            # Create the plot with a dynamic height  
            fig, ax = plt.subplots(figsize=(4, height))  # Width is 5, height is dynamic
            
            # Convert category labels if value labels exist  
            if var in value_labels:  
                freq.index = [value_labels[var].get(val, val) for val in freq.index]  

            # Wrap y-axis labels if they exceed 20 characters  
            wrapped_labels = [textwrap.fill(str(label), width=20) for label in freq.index]  

            # Create horizontal bar plot  
            bars = ax.barh(range(len(freq)), freq.values, color="skyblue", edgecolor="black")  

            # Add padding to the x-axis to create space for labels  
            max_freq = max(freq.values)  
            ax.set_xlim(0, max_freq * 1.2)  # 50% padding on the right  

            # Calculate a proportional offset for outside labels (10% of max_freq)  
            outside_offset = max_freq * 0.03  # Proportional offset to ensure visibility  
            if outside_offset < 1:  # Ensure a minimum offset for small scales  
                outside_offset = 1  

            # Add percentage labels, dynamically positioning to avoid border overlap  
            for i, (v, p) in enumerate(zip(freq.values, percentages)):  
                # Calculate the threshold for placing the label inside (e.g., within 5% of the max x-limit)  
                xlim_max = ax.get_xlim()[1]  
                margin = 0.05 * xlim_max  # 5% of the x-axis max as the margin  
                if v > xlim_max - margin:  
                    # Place inside the bar, align right, white text for readability  
                    ax.text(v - 1.5, i, f'{p:.1f}% ', ha='right', va='center', fontsize=7, color='white')  
                else:  
                    # Place outside the bar with a proportional offset  
                    label_pos = v + outside_offset  
                    print(f"Label position (outside): {label_pos}")  
                    ax.text(label_pos, i, f'{p:.1f}%', va='center', fontsize=7)  

            # Set title and axis labels with reduced font size  
            ax.set_title(f"Weighted Frequencies for {label}", fontsize=7)  
            ax.set_xlabel("Weighted Count", fontsize=7)  
            ax.set_yticks(range(len(freq)))  
            ax.set_yticklabels(wrapped_labels, fontsize=7)  # Use wrapped labels with reduced font size  
            ax.tick_params(axis='x', labelsize=7)  # Reduced font size for x-tick labels  

            # Adjust plot margins to ensure space for labels  
            plt.subplots_adjust(right=0.85)  # Leave 15% space on the right for labels
            
            # Use st.columns to make the plot half the window width  
            col1, col2 = st.columns(2)  # Create two equal columns (each 50% of the window width)  
            with col1:  
                st.pyplot(fig)  # Plot in the first column, which is half the window width

            # Close the figure to free memory  
            plt.close(fig)
            
# Instructions section
st.header("Instructions")
st.markdown("""
- **Select Variables**: Choose any variables - continous and categorical variables will be handled differently.
- **View Results**: Results are displayed in an APA-style table with with relevant statistics that describe the sample. You can copy and past the table directly into a Word document.
- **Review visualsation**: Bar charts, histograms, and box plots are provided as appropriate.
""")