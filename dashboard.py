#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Page config
st.set_page_config(
    page_title="Epsilon Health analytics dashboard",
    layout="wide"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global font family */
    html, body, [class*="css"]  {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Main title styling */
    h1 {
        font-weight: 600 !important;
        color: #1e293b !important;
        letter-spacing: -0.02em !important;
    }
    
    /* Headers styling */
    h2, h3 {
        font-weight: 500 !important;
        color: #334155 !important;
        letter-spacing: -0.01em !important;
    }
    
    /* Metrics styling */
    [data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #fed7aa;
        padding: 16px;
        border-radius: 8px;
        box-shadow: 0 1px 3px 0 rgba(251, 146, 60, 0.1);
    }
    
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-weight: 600;
        color: #ea580c;
    }
    
    [data-testid="metric-container"] [data-testid="stMetricLabel"] {
        color: #64748b;
    }
    
    /* Primary button - orange accent */
    .stButton > button {
        background-color: #fb923c;
        color: white;
        border: none;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #f97316;
        box-shadow: 0 4px 6px -1px rgba(251, 146, 60, 0.3);
    }
    
    /* Sidebar specific buttons and elements with orange */
    [data-testid="stSidebar"] .stButton > button {
        background-color: #fb923c;
        color: white;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #f97316;
    }
    
    /* Radio button selected state - orange */
    [data-testid="stSidebar"] input[type="radio"]:checked + div {
        color: #fb923c;
        font-weight: 500;
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Selectbox and multiselect styling */
    .stSelectbox > div > div, .stMultiSelect > div > div {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        font-weight: 400;
    }
    
    /* Multiselect selected items - orange theme */
    .stMultiSelect > div > div > div[data-baseweb="tag"] {
        background-color: #fb923c !important;
    }
    
    /* Multiselect dropdown hover */
    .stMultiSelect div[role="option"]:hover {
        background-color: #fed7aa !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 2px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: transparent;
        border: none;
        color: #64748b;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        border-bottom: 3px solid #fb923c;
        color: #fb923c;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Success/Info/Warning boxes */
    .stSuccess, .stInfo, .stWarning {
        border-radius: 8px;
        font-weight: 400;
    }
    
    /* Download button with orange accent */
    .stDownloadButton > button {
        background-color: #fb923c;
        color: white;
        border: none;
        font-weight: 500;
    }
    
    .stDownloadButton > button:hover {
        background-color: #f97316;
    }
    
    /* Divider styling */
    hr {
        border-color: #e2e8f0;
        margin: 24px 0;
    }
    
    /* File uploader */
    .stFileUploader > div {
        border: 2px dashed #cbd5e1;
        border-radius: 8px;
        background-color: #f8fafc;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background-color: transparent;
    }
    
    /* Expander headers */
    .streamlit-expanderHeader {
        font-weight: 500;
        color: #334155;
    }
    
    /* Make plotly charts match theme */
    .js-plotly-plot .plotly .gtitle {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("Epsilon Health analytics dashboard")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'filtered_df' not in st.session_state:
    st.session_state.filtered_df = None

# Sidebar for file selection and filters
with st.sidebar:
    st.header("Data Selection")
    
    df = None
    
    uploaded_file = st.file_uploader("Choose a processed CSV file", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df)} rows")
    
    if df is not None:
        st.session_state.df = df
        
        # Convert datetime columns if they exist
        datetime_columns = [col for col in df.columns if 'datetime' in col.lower() or 'date' in col.lower()]
        for col in datetime_columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
        
        st.divider()
        st.header("Filters")
        
        # Date filter
        date_columns = [col for col in df.columns if 'datetime' in col.lower() or 'completed_at' in col.lower()]
        if date_columns:
            # Prefer 'datetime' column if it exists, otherwise use first available
            if 'datetime' in date_columns:
                default_date_col = 'datetime'
            else:
                default_date_col = date_columns[0]
            
            date_col = st.selectbox("Date column:", date_columns, index=date_columns.index(default_date_col))
            
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                min_date = df[date_col].min()
                max_date = df[date_col].max()
                
                if pd.notna(min_date) and pd.notna(max_date):
                    date_range = st.date_input(
                        "Date range:",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date
                    )
                    
                    if len(date_range) == 2:
                        start_date, end_date = date_range
                        start_datetime = pd.Timestamp(start_date)
                        end_datetime = pd.Timestamp(end_date) + timedelta(days=1)
                        mask = (df[date_col] >= start_datetime) & (df[date_col] < end_datetime)
                        filtered_df = df[mask].copy()
                    else:
                        filtered_df = df.copy()
                else:
                    filtered_df = df.copy()
                    st.info("No valid dates found in selected column")
        else:
            filtered_df = df.copy()
            st.info("No datetime columns found")
        
        # Model version filter
        version_col = "generation_version"
        
        print(f"version col = {version_col}")
        if version_col:
            versions = df[version_col].dropna().unique()  # Use original df to get all versions
            selected_versions = st.multiselect(
                f"Filter by model version ({version_col}):",
                options=sorted(versions),  # Sort for better display
                default=list(versions)  # Convert to list
            )
            if selected_versions:
                filtered_df = filtered_df[filtered_df[version_col].isin(selected_versions)]
        
        # Radiologist filter
        if 'radiologist_first_name' in df.columns and 'radiologist_last_name' in df.columns:
            filtered_df['radiologist_name'] = filtered_df['radiologist_first_name'].fillna('') + ' ' + filtered_df['radiologist_last_name'].fillna('')
            radiologists = filtered_df['radiologist_name'].str.strip().unique()
            radiologists = [r for r in radiologists if r]
            
            selected_radiologists = st.multiselect(
                "Filter by radiologist:",
                options=radiologists,
                default=radiologists
            )
            if selected_radiologists:
                filtered_df = filtered_df[filtered_df['radiologist_name'].isin(selected_radiologists)]
        
        st.session_state.filtered_df = filtered_df
        
        # Show filter summary
        st.divider()
        st.metric("Filtered rows", f"{len(filtered_df):,} / {len(df):,}")
        
        # Session break duration setting
        st.divider()
        st.subheader("Session Analysis Settings")
        session_break_minutes = st.slider(
            "Session break threshold (minutes)",
            min_value=1,
            max_value=30,
            value=10,
            step=1,
            help="Minutes of inactivity to consider a new reading session"
        )
        
        # Store in session state for access in tabs
        st.session_state.session_break_minutes = session_break_minutes

# Main dashboard area
if st.session_state.filtered_df is not None:
    df = st.session_state.filtered_df
    
    # Calculate metrics
    total_reports = len(df)
    
    # Skip addendums for metrics - use .copy() to avoid the warning
    if 'is_addendum' in df.columns:
        metrics_df = df[df['is_addendum'].astype(str).str.lower() != 'true'].copy()
    else:
        metrics_df = df.copy()
    
    # Calculate accuracy metrics
    if 'is_correct' in df.columns:
        correct_count = (metrics_df['is_correct'].astype(str).str.lower() == 'true').sum()
        accuracy = (correct_count / len(metrics_df) * 100) if len(metrics_df) > 0 else 0
    else:
        accuracy = None
    
    if 'is_normal' in df.columns:
        normal_count = (metrics_df['is_normal'].astype(str).str.lower() == 'true').sum()
        normal_rate = (normal_count / len(metrics_df) * 100) if len(metrics_df) > 0 else 0
    else:
        normal_rate = None
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reports", f"{total_reports:,}")
    
    with col2:
        if accuracy is not None:
            st.metric("Overall Accuracy", f"{accuracy:.1f}%")
        else:
            st.metric("Overall Accuracy", "N/A")
    
    with col3:
        if normal_rate is not None:
            st.metric("Normal Report Rate", f"{normal_rate:.1f}%")
        else:
            st.metric("Normal Report Rate", "N/A")
    
    with col4:
        if 'is_addendum' in df.columns:
            addendum_count = (df['is_addendum'].astype(str).str.lower() == 'true').sum()
            st.metric("Addendums", f"{addendum_count:,}")
        else:
            st.metric("Addendums", "N/A")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Study Breakdown", "Radiologist Performance", "Reading Sessions", "Raw Data"])
    
    with tab1:
        st.header("Overview")
        
        # Contingency Matrices Section
        st.subheader("Contingency Matrices")
        
        # Helper function to create contingency matrix
        def create_contingency_matrix(df_input, correct_column):
            # Make a copy to avoid modifying original
            df = df_input.copy()
            
            # Skip addendums if column exists
            if 'is_addendum' in df.columns:
                df = df[df['is_addendum'].astype(str).str.lower() != 'true']
            
            # Check if required columns exist
            if 'is_normal' not in df.columns or correct_column not in df.columns:
                return None, None
            
            # Create contingency matrix manually
            matrix = np.zeros((2, 2), dtype=int)
            
            for _, row in df.iterrows():
                # Handle different data types and skip invalid rows
                try:
                    normal_val = str(row['is_normal']).strip().lower()
                    correct_val = str(row[correct_column]).strip().lower()
                    
                    # Skip rows that don't have valid true/false values
                    if normal_val not in ['true', 'false'] or correct_val not in ['true', 'false']:
                        continue
                    
                    # Skip n/a values
                    if normal_val == 'n/a' or correct_val == 'n/a':
                        continue
                    
                    is_normal = normal_val == 'true'
                    is_correct = correct_val == 'true'
                    
                    # matrix[normal_idx][correct_idx]
                    # [0,0] = not normal, not correct
                    # [0,1] = not normal, correct
                    # [1,0] = normal, not correct  
                    # [1,1] = normal, correct
                    normal_idx = 1 if is_normal else 0
                    correct_idx = 1 if is_correct else 0
                    matrix[normal_idx, correct_idx] += 1
                    
                except:
                    # Skip any problematic rows
                    continue
            
            # Check if we have any data
            total = matrix.sum()
            if total == 0:
                return None, None
            
            # Calculate percentages
            percentages = (matrix / total * 100).round(1)
            
            return matrix, percentages
        
        # Create three columns for the matrices
        matrix_cols = st.columns(3)
        
        # Matrix 1: Findings + Impressions (is_correct)
        with matrix_cols[0]:
            if 'is_correct' in df.columns:
                st.markdown("**Findings + Impressions Match**")
                matrix, percentages = create_contingency_matrix(metrics_df, 'is_correct')
                
                if matrix is not None:
                    # Create display dataframe with counts and percentages
                    display_data = []
                    for normal_idx in [0, 1]:
                        row_data = []
                        for correct_idx in [0, 1]:
                            count = int(matrix[normal_idx, correct_idx])
                            row_data.append(str(count))
                        display_data.append(row_data)
                    
                    # Create styled table
                    matrix_df = pd.DataFrame(
                        display_data,
                        index=['Abnormal', 'Normal'],
                        columns=['Incorrect', 'Correct']
                    )
                    
                    # Don't set index/column names to avoid extra header row
                    # matrix_df.index.name = 'Normal'
                    # matrix_df.columns.name = 'Correct'
                    
                    # Generate HTML
                    raw_html = matrix_df.to_html(escape=False, index=True, header=True)
                    
                    # Just display the raw HTML without styling since that works
                    st.markdown(raw_html, unsafe_allow_html=True)
                    
                    # Calculate and display correlation
                    if matrix.shape == (2, 2):
                        # Matthews Correlation Coefficient
                        tn = matrix[0, 0]
                        fp = matrix[0, 1]
                        fn = matrix[1, 0]
                        tp = matrix[1, 1]
                        
                        numerator = (tp * tn) - (fp * fn)
                        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                        mcc = numerator / denominator if denominator != 0 else 0
                        
                        st.caption(f"MCC: {mcc:.3f}")
                else:
                    st.info("No data available")
            else:
                st.info("is_correct column not found")
        
        # Matrix 2: Findings only
        with matrix_cols[1]:
            if 'is_findings_correct' in df.columns:
                st.markdown("**Findings Match Only**")
                matrix, percentages = create_contingency_matrix(metrics_df, 'is_findings_correct')
                
                if matrix is not None:
                    # Create display dataframe with counts and percentages
                    display_data = []
                    for normal_idx in [0, 1]:
                        row_data = []
                        for correct_idx in [0, 1]:
                            count = int(matrix[normal_idx, correct_idx])
                            row_data.append(str(count))
                        display_data.append(row_data)
                    
                    # Create styled table
                    matrix_df = pd.DataFrame(
                        display_data,
                        index=['Abnormal', 'Normal'],
                        columns=['Incorrect', 'Correct']
                    )
                    
                    # Don't set index/column names to avoid extra header row
                    
                    # Generate HTML
                    raw_html = matrix_df.to_html(escape=False, index=True, header=True)
                    
                    # Just display the raw HTML without styling since that works
                    st.markdown(raw_html, unsafe_allow_html=True)
                    
                    # Calculate MCC
                    if matrix.shape == (2, 2):
                        tn = matrix[0, 0]
                        fp = matrix[0, 1]
                        fn = matrix[1, 0]
                        tp = matrix[1, 1]
                        
                        numerator = (tp * tn) - (fp * fn)
                        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                        mcc = numerator / denominator if denominator != 0 else 0
                        
                        st.caption(f"MCC: {mcc:.3f}")
                else:
                    st.info("No data available")
            else:
                st.info("is_findings_correct column not found")
        
        # Matrix 3: Impressions only
        with matrix_cols[2]:
            if 'is_impressions_correct' in df.columns:
                st.markdown("**Impressions Match Only**")
                matrix, percentages = create_contingency_matrix(metrics_df, 'is_impressions_correct')
                
                if matrix is not None:
                    # Create display dataframe with counts and percentages
                    display_data = []
                    for normal_idx in [0, 1]:
                        row_data = []
                        for correct_idx in [0, 1]:
                            count = int(matrix[normal_idx, correct_idx])
                            row_data.append(str(count))
                        display_data.append(row_data)
                    
                    # Create styled table
                    matrix_df = pd.DataFrame(
                        display_data,
                        index=['Abnormal', 'Normal'],
                        columns=['Incorrect', 'Correct']
                    )
                    
                    # Don't set index/column names to avoid extra header row
                    
                    # Generate HTML
                    raw_html = matrix_df.to_html(escape=False, index=True, header=True)
                    
                    # Just display the raw HTML without styling since that works
                    st.markdown(raw_html, unsafe_allow_html=True)
                    
                    # Calculate MCC
                    if matrix.shape == (2, 2):
                        tn = matrix[0, 0]
                        fp = matrix[0, 1]
                        fn = matrix[1, 0]
                        tp = matrix[1, 1]
                        
                        numerator = (tp * tn) - (fp * fn)
                        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                        mcc = numerator / denominator if denominator != 0 else 0
                        
                        st.caption(f"MCC: {mcc:.3f}")
                else:
                    st.info("No data available")
            else:
                st.info("is_impressions_correct column not found")
        
        st.divider()
        
        # Additional metrics if available
        if 'is_findings_correct' in df.columns and 'is_impressions_correct' in df.columns:
            st.subheader("Detailed Accuracy Breakdown")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                findings_correct = (metrics_df['is_findings_correct'].astype(str).str.lower() == 'true').sum()
                findings_acc = (findings_correct / len(metrics_df) * 100) if len(metrics_df) > 0 else 0
                st.metric("Findings Accuracy", f"{findings_acc:.1f}%")
            
            with col2:
                impressions_correct = (metrics_df['is_impressions_correct'].astype(str).str.lower() == 'true').sum()
                impressions_acc = (impressions_correct / len(metrics_df) * 100) if len(metrics_df) > 0 else 0
                st.metric("Impressions Accuracy", f"{impressions_acc:.1f}%")
            
            with col3:
                only_impressions_wrong = ((metrics_df['is_findings_correct'].astype(str).str.lower() == 'true') & 
                                         (metrics_df['is_correct'].astype(str).str.lower() == 'false')).sum()
                st.metric("Only Impressions Changed", f"{only_impressions_wrong:,}")
    
    with tab2:
        st.header("Study Type Breakdown")
        
        if 'study_description' in df.columns:
            # Normalize study descriptions (remove LEFT/RIGHT)
            def normalize_study(desc):
                if pd.isna(desc):
                    return desc
                desc = str(desc).strip()
                if desc.upper().endswith('(LEFT)'):
                    return desc[:-6].strip()
                elif desc.upper().endswith('(RIGHT)'):
                    return desc[:-7].strip()
                return desc
            
            metrics_df['study_normalized'] = metrics_df['study_description'].apply(normalize_study)
            
            # Group by study type
            study_stats = []
            for study in metrics_df['study_normalized'].dropna().unique():
                study_data = metrics_df[metrics_df['study_normalized'] == study]
                
                stats = {
                    'Study Type': study,
                    'Total': len(study_data),
                    'Normal': (study_data['is_normal'].astype(str).str.lower() == 'true').sum() if 'is_normal' in df.columns else 0,
                    'Correct': (study_data['is_correct'].astype(str).str.lower() == 'true').sum() if 'is_correct' in df.columns else 0
                }
                
                stats['% Normal'] = (stats['Normal'] / stats['Total'] * 100) if stats['Total'] > 0 else 0
                stats['% Correct'] = (stats['Correct'] / stats['Total'] * 100) if stats['Total'] > 0 else 0
                
                study_stats.append(stats)
            
            study_df = pd.DataFrame(study_stats).sort_values('Total', ascending=False)
            
            # Table
            st.subheader("Study Statistics")
            # Dynamic height: 35px per row + 50px for header, max 600px
            table_height = min(600, (len(study_df) * 35) + 50)
            st.dataframe(
                study_df.style.format({
                    '% Normal': '{:.1f}%',
                    '% Correct': '{:.1f}%'
                }),
                width='stretch',
                height=table_height
            )
    
    with tab3:
        st.header("Radiologist Performance")
        
        if 'radiologist_name' in df.columns or ('radiologist_first_name' in df.columns and 'radiologist_last_name' in df.columns):
            if 'radiologist_name' not in df.columns:
                metrics_df['radiologist_name'] = metrics_df['radiologist_first_name'].fillna('') + ' ' + metrics_df['radiologist_last_name'].fillna('')
            
            # Calculate radiologist stats
            rad_stats = []
            for rad in metrics_df['radiologist_name'].dropna().unique():
                if not rad.strip():
                    continue
                    
                rad_data = metrics_df[metrics_df['radiologist_name'] == rad]
                
                stats = {
                    'Radiologist': rad,
                    'Total Reports': len(rad_data),
                    'Normal': (rad_data['is_normal'].astype(str).str.lower() == 'true').sum() if 'is_normal' in df.columns else 0,
                    'Correct': (rad_data['is_correct'].astype(str).str.lower() == 'true').sum() if 'is_correct' in df.columns else 0
                }
                
                stats['% Normal'] = (stats['Normal'] / stats['Total Reports'] * 100) if stats['Total Reports'] > 0 else 0
                stats['% Correct'] = (stats['Correct'] / stats['Total Reports'] * 100) if stats['Total Reports'] > 0 else 0
                
                rad_stats.append(stats)
            
            rad_df = pd.DataFrame(rad_stats).sort_values('Total Reports', ascending=False)
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Radiologists", len(rad_df))
            with col2:
                avg_accuracy = rad_df['% Correct'].mean()
                st.metric("Average Accuracy", f"{avg_accuracy:.1f}%")
            with col3:
                std_accuracy = rad_df['% Correct'].std()
                st.metric("Std Dev Accuracy", f"{std_accuracy:.1f}%")
            
            # Table
            st.subheader("Detailed Radiologist Statistics")
            # Dynamic height: 35px per row + 50px for header, max 500px
            table_height = min(500, (len(rad_df) * 35) + 50)
            st.dataframe(
                rad_df.style.format({
                    '% Normal': '{:.1f}%',
                    '% Correct': '{:.1f}%'
                }),
                width='stretch',
                height=table_height
            )
    
    with tab4:
        st.header("Reading Sessions Analysis")
        
        # Get session break duration from session state
        session_break_minutes = st.session_state.get('session_break_minutes', 30)
        
        # Use submitted_datetime column (the Unix timestamp column)
        timestamp_col = 'submitted_datetime'
        
        if timestamp_col not in df.columns:
            st.warning(f"'{timestamp_col}' column not found. Reading session analysis requires timestamp data.")
        else:
            # Prepare data for session analysis
            analysis_df = metrics_df.copy()
            
            # Convert Unix timestamp to datetime
            # FORCE conversion from numeric timestamp - the column contains Unix timestamps as numbers
            # First, ensure the column is numeric
            analysis_df[timestamp_col] = pd.to_numeric(analysis_df[timestamp_col], errors='coerce')
            
            # Get a sample value to determine the unit
            sample_value = analysis_df[timestamp_col].iloc[0] if len(analysis_df) > 0 else None
            
            # These should be Unix timestamps in seconds (e.g., 1755716143 = year 2025)
            # Convert from seconds
            analysis_df['datetime_converted'] = pd.to_datetime(analysis_df[timestamp_col], unit='s', errors='coerce')
            
            datetime_col = 'datetime_converted'
            analysis_df = analysis_df[analysis_df[datetime_col].notna()]
            
            if len(analysis_df) == 0:
                st.warning("No valid datetime data found for analysis.")
            else:
                # Function to format duration
                def format_duration(seconds):
                    if seconds < 60:
                        return f"{seconds:.1f}s"
                    elif seconds < 3600:
                        return f"{seconds/60:.1f}m"
                    else:
                        return f"{seconds/3600:.1f}h"
                
                # Analyze sessions by radiologist
                radiologist_col = 'radiologist_name' if 'radiologist_name' in analysis_df.columns else None
                
                if not radiologist_col:
                    st.info("No radiologist data found. Analyzing all reports together.")
                    analysis_df['radiologist_name'] = 'All Reports'
                    radiologist_col = 'radiologist_name'
                
                # Group by radiologist and sort by datetime
                all_reading_times = []
                all_sessions = []
                radiologist_stats = {}
                
                for radiologist in analysis_df[radiologist_col].unique():
                    if not radiologist or pd.isna(radiologist):
                        continue
                    
                    rad_df = analysis_df[analysis_df[radiologist_col] == radiologist].sort_values(datetime_col)
                    
                    if len(rad_df) < 2:
                        continue
                    
                    # Calculate time intervals between consecutive reports
                    rad_df = rad_df.copy()  # Avoid SettingWithCopyWarning
                    rad_df.loc[:, 'time_diff'] = rad_df[datetime_col].diff().dt.total_seconds()
                    
                    # Identify sessions based on break threshold
                    session_break_seconds = session_break_minutes * 60
                    rad_df.loc[:, 'new_session'] = rad_df['time_diff'] > session_break_seconds
                    rad_df.loc[:, 'session_id'] = rad_df['new_session'].cumsum()
                    
                    # Calculate reading times (exclude session breaks)
                    max_reading_seconds = 20 * 60  # 20 minutes max reasonable reading time
                    valid_times = rad_df[(rad_df['time_diff'] > 0) & 
                                        (rad_df['time_diff'] <= max_reading_seconds) & 
                                        (~rad_df['new_session'])]['time_diff'].values
                    
                    if len(valid_times) > 0:
                        all_reading_times.extend(valid_times)
                        
                        # Calculate statistics for this radiologist
                        radiologist_stats[radiologist] = {
                            'total_reports': len(rad_df),
                            'valid_reading_times': len(valid_times),
                            'mean_seconds': np.mean(valid_times),
                            'median_seconds': np.median(valid_times),
                            'std_seconds': np.std(valid_times) if len(valid_times) > 1 else 0,
                            'min_seconds': np.min(valid_times),
                            'max_seconds': np.max(valid_times),
                            'sessions': rad_df['session_id'].nunique()
                        }
                        
                        # Analyze by correctness if available
                        if 'is_correct' in rad_df.columns:
                            correct_times = rad_df[(rad_df['time_diff'] > 0) & 
                                                  (rad_df['time_diff'] <= max_reading_seconds) & 
                                                  (~rad_df['new_session']) & 
                                                  (rad_df['is_correct'].astype(str).str.lower() == 'true')]['time_diff'].values
                            incorrect_times = rad_df[(rad_df['time_diff'] > 0) & 
                                                    (rad_df['time_diff'] <= max_reading_seconds) & 
                                                    (~rad_df['new_session']) & 
                                                    (rad_df['is_correct'].astype(str).str.lower() == 'false')]['time_diff'].values
                            
                            if len(correct_times) > 0:
                                radiologist_stats[radiologist]['correct_mean'] = np.mean(correct_times)
                                radiologist_stats[radiologist]['correct_median'] = np.median(correct_times)
                            
                            if len(incorrect_times) > 0:
                                radiologist_stats[radiologist]['incorrect_mean'] = np.mean(incorrect_times)
                                radiologist_stats[radiologist]['incorrect_median'] = np.median(incorrect_times)
                
                if len(all_reading_times) > 0:
                    # Overall statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Average Reading Time", format_duration(np.mean(all_reading_times)))
                    with col2:
                        st.metric("Median Reading Time", format_duration(np.median(all_reading_times)))
                    with col3:
                        st.metric("Total Valid Readings", f"{len(all_reading_times):,}")
                    with col4:
                        st.metric("Total Radiologists", len(radiologist_stats))
                    
                    # Distribution histogram
                    st.subheader("Reading Time Distribution")
                    
                    # Convert to minutes for display
                    times_minutes = [t/60 for t in all_reading_times]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=times_minutes,
                        nbinsx=50,
                        name='Reading Times',
                        marker_color='#fb923c'
                    ))
                    
                    # Add mean and median lines
                    mean_minutes = np.mean(times_minutes)
                    median_minutes = np.median(times_minutes)
                    
                    fig.add_vline(x=mean_minutes, line_dash="dash", line_color="red", 
                                annotation_text=f"Mean:\n{mean_minutes:.1f}m")
                    fig.add_vline(x=median_minutes, line_dash="dash", line_color="green",
                                annotation_text=f"Median:\n{median_minutes:.1f}m")
                    
                    fig.update_layout(
                        title="Distribution of Reading Times",
                        xaxis_title="Reading Time (minutes)",
                        yaxis_title="Frequency",
                        showlegend=False,
                        height=400,
                        font=dict(family="Inter, sans-serif"),
                        plot_bgcolor='white',
                        paper_bgcolor='white'
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                    
                    # Time distribution breakdown
                    st.subheader("Time Distribution Breakdown")
                    
                    bins = [0, 30, 60, 120, 300, 600, float('inf')]
                    bin_labels = ["<30s", "30s-1m", "1-2m", "2-5m", "5-10m", ">10m"]
                    bin_counts = pd.cut(all_reading_times, bins=bins, labels=bin_labels).value_counts()
                    
                    # Create distribution chart
                    fig_dist = go.Figure(data=[
                        go.Bar(
                            x=bin_counts.index,
                            y=bin_counts.values,
                            text=[f"{v} ({v/len(all_reading_times)*100:.1f}%)" for v in bin_counts.values],
                            textposition='auto',
                            marker_color='#fb923c'
                        )
                    ])
                    
                    fig_dist.update_layout(
                        title="Reading Time Categories",
                        xaxis_title="Time Range",
                        yaxis_title="Count",
                        height=350,
                        font=dict(family="Inter, sans-serif"),
                        plot_bgcolor='white',
                        paper_bgcolor='white'
                    )
                    
                    st.plotly_chart(fig_dist, width='stretch')
                    
                    # Radiologist breakdown
                    if len(radiologist_stats) > 1 or (len(radiologist_stats) == 1 and 'All Reports' not in radiologist_stats):
                        st.subheader("Performance by Radiologist")
                        
                        # Create dataframe for display
                        rad_data = []
                        for rad, stats in radiologist_stats.items():
                            row = {
                                'Radiologist': rad,
                                'Reports': stats['total_reports'],
                                'Valid Times': stats['valid_reading_times'],
                                'Sessions': stats['sessions'],
                                'Avg Time': format_duration(stats['mean_seconds']),
                                'Median Time': format_duration(stats['median_seconds']),
                                'Std Dev': format_duration(stats['std_seconds'])
                            }
                            
                            # Add correct/incorrect if available
                            if 'correct_mean' in stats:
                                row['Avg Correct'] = format_duration(stats['correct_mean'])
                            if 'correct_median' in stats:
                                row['Median Correct'] = format_duration(stats['correct_median'])
                            if 'incorrect_mean' in stats:
                                row['Avg Incorrect'] = format_duration(stats['incorrect_mean'])
                            if 'incorrect_median' in stats:
                                row['Median Incorrect'] = format_duration(stats['incorrect_median'])
                            
                            rad_data.append(row)
                        
                        rad_df = pd.DataFrame(rad_data).sort_values('Reports', ascending=False)
                        
                        # Display table
                        st.dataframe(rad_df, use_container_width=True)
                    
                    # Percentiles
                    st.subheader("Reading Time Percentiles")
                    
                    percentiles = [10, 25, 50, 75, 90, 95, 99]
                    perc_values = np.percentile(all_reading_times, percentiles)
                    
                    perc_df = pd.DataFrame({
                        'Percentile': [f"{p}th" for p in percentiles],
                        'Time': [format_duration(v) for v in perc_values]
                    })
                    
                    st.dataframe(perc_df, use_container_width=True)
                
                else:
                    st.warning("No valid reading time data found. Check that reports have proper timestamps and are from the same radiologist.")
    
    with tab5:
        st.header("Raw Data Explorer")
        
        # Search box
        search = st.text_input("Search in data (searches all columns):", "")
        
        display_df = st.session_state.filtered_df.copy()
        
        # Sort by datetime if it exists
        if 'datetime' in display_df.columns:
            display_df = display_df.sort_values('datetime', ascending=False)
        
        if search:
            mask = display_df.astype(str).apply(lambda x: x.str.contains(search, case=False, na=False)).any(axis=1)
            display_df = display_df[mask]
            st.info(f"Found {len(display_df)} matching rows")
        
        # Column selector
        all_columns = display_df.columns.tolist()
        # Filter default columns - include datetime but exclude submitted_datetime and claimed_datetime
        default_columns = []
        for col in all_columns:
            col_lower = col.lower()
            # Skip submitted_datetime and claimed_datetime
            if 'submitted_datetime' in col_lower or 'claimed_datetime' in col_lower:
                continue
            # Include columns with these keywords
            if any(x in col_lower for x in ['study', 'normal', 'correct', 'radiologist', 'version']):
                default_columns.append(col)
            # Special case: include 'datetime' column exactly
            elif col == 'datetime':
                default_columns.append(col)
        
        selected_columns = st.multiselect(
            "Select columns to display:",
            options=all_columns,
            default=default_columns[:10] if default_columns else all_columns[:10]
        )
        
        if selected_columns:
            # Dynamic height: 35px per row + 50px for header, max 600px
            table_height = min(600, (len(display_df) * 35) + 50)
            st.dataframe(display_df[selected_columns], use_container_width=True, height=table_height)
            
            # Export functionality
            csv = display_df[selected_columns].to_csv(index=False)
            st.download_button(
                label="Download filtered data as CSV",
                data=csv,
                file_name=f"filtered_radiology_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("Please select at least one column to display")

else:
    st.info("Please load a processed CSV file from the sidebar to begin")
    
    st.subheader("Expected CSV columns:")
    st.markdown("""
    Your processed CSV should contain these columns:
    - **datetime columns**: `datetime` or similar
    - **is_normal**: Whether the radiologist report is normal
    - **is_ai_report_normal**: Whether the AI report is normal
    - **is_correct**: Whether AI matches radiologist
    - **is_findings_correct**: Whether findings section matches
    - **is_impressions_correct**: Whether impressions section matches
    - **study_description**: Type of X-ray
    - **radiologist_first_name** / **radiologist_last_name**: Radiologist info
    - **generation_version**: AI model version
    """)