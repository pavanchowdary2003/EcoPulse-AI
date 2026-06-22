import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import base64
import os
import gdown

# 1. Initialize Page Geometry Configuration
st.set_page_config(
    page_title="Dataset Insights",
    page_icon="⚡",
    layout="wide"
)

# 2. Automotive Background Engine
def set_cluster_background(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpeg;base64,{encoded_string}");
                background-size: cover;
                background-position: center center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            
            /* COMPACT GLASSMORPHISM CONTAINERS */
            div[data-testid="stDataFrame"], div[data-testid="stVerticalBlock"] > div, .stPlotlyChart, .element-container {{
                background-color: rgba(15, 23, 42, 0.75) !important;
                border-radius: 8px;
                padding: 16px !important;
                margin-bottom: 8px !important;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
                backdrop-filter: blur(6px);
                -webkit-backdrop-filter: blur(6px);
                border: 1px solid rgba(255, 255, 255, 0.12);
            }}
            
            h1, h2, h3, h4, h5, p, span, label, div, [data-testid="stMarkdownText"] {{
                color: #ffffff !important;
            }}
            
            h1 {{
                font-size: 1.8rem !important;
                margin-top: 0px !important;
            }}
            h3 {{
                font-size: 1.25rem !important;
                font-weight: 600 !important;
                margin-top: 10px !important;
                border-left: 3px solid #38bdf8;
                padding-left: 8px;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

set_cluster_background("assets/unnamed.jpg")

# 3. Premium Compact Title Banner
logo_col, title_col = st.columns([1, 12])
with logo_col:
    st.markdown("<h1 style='margin-top: 0px; padding-top: 0px; text-align: center;'>🚗</h1>", unsafe_allow_html=True)
with title_col:
    st.markdown("<h1 style='margin-bottom: 0px; padding-top: 0px;'>Dataset Insights Dashboard</h1>", unsafe_allow_html=True)

st.markdown("---")

# 4. Large-File Cloud Streaming Loader Engine
FILE_ID = "1zyZ6WvdhxXGKH3JdWx2xHXauu-NyXMRv"

@st.cache_data(show_spinner="Downloading and caching engine dataset from cloud storage (109 MB)...")
def load_large_drive_dataset(file_id):
    local_output = "data/cached_engine1.csv"
    
    # Create target folder structural path if missing
    os.makedirs("data", exist_ok=True)
    
    # If the file hasn't been cached locally yet, download it directly bypassing checks
    if not os.path.exists(local_output):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, local_output, quiet=True)
        
    try:
        return pd.read_csv(local_output)
    except Exception as e:
        st.error(f"Error parsing dataset file structure: {e}")
        return None

# Execution happens safely AFTER the function is declared
df = load_large_drive_dataset(FILE_ID)

# 5. Render Analytical Dashboard Content
if df is not None and not df.empty:
    # Split Row 1: Metrics and Dataset Dimensions
    c1, c2 = st.columns([1, 3])
    with c1:
        st.subheader("Dataset Shape")
        st.metric(label="Total Logged Rows", value=f"{df.shape[0]:,}")
        st.metric(label="Telemetry Attributes", value=str(df.shape[1]))
    with c2:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(5), use_container_width=True)

    # Row 2: Comprehensive Statistical Metrics Summary
    st.subheader("Engine Parameter Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    # Row 3: Heatmap Plotting Block
    st.subheader("Cross-Parameter Correlation Analysis")
    numeric_df = df.select_dtypes(include=[np.number])
    
    if not numeric_df.empty:
        corr_matrix = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(12, 8), dpi=110)
        fig.patch.set_facecolor('none')  
        ax.set_facecolor('none')

        sns.heatmap(
            corr_matrix,
            cmap="viridis",
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            linecolor="#1e293b", 
            square=True,
            cbar_kws={"shrink": 0.75, "label": "Correlation Strength (-1 to +1)"},
            ax=ax
        )

        ax.set_title("Correlation Matrix – Engine 1 Co-Dependencies", fontsize=14, color="#ffffff", fontweight="bold", pad=15)
        ax.tick_params(colors='#ffffff', labelsize=9)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", color="#cbd5e1")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, color="#cbd5e1")
        
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color='#ffffff', labelcolor='#ffffff')
        
        plt.tight_layout()
        st.pyplot(fig)
else:
    st.error("⚠️ Failed to load dataset. Please double-check file configuration options.")