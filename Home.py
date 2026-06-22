import streamlit as st
import base64
import os

# 1. Initialize Streamlit Page Geometry Configurations
st.set_page_config(
    page_title="EcoPulse-AI Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Automotive Instrument Cluster Background Engine
def set_cluster_background(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        
        # Inject structural layout CSS with significantly smaller content blocks
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
            
            /* COMPACT GLASSMORPHISM OVERLAYS - Reduced padding and constraints */
            [data-testid="stMetric"], div[data-testid="stVerticalBlock"] > div, .stAlert {{
                background-color: rgba(15, 23, 42, 0.75) !important;
                border-radius: 8px;
                padding: 10px 14px !important; /* Shrunk padding */
                margin-bottom: 8px !important;   /* Tighter grid spacing */
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
                backdrop-filter: blur(6px);
                -webkit-backdrop-filter: blur(6px);
                border: 1px solid rgba(255, 255, 255, 0.12);
            }}
            
            /* High-contrast colors for typography text */
            h1, h2, h3, p, span, label, div {{
                color: #ffffff !important;
            }}
            
            /* Scaled Down Compact Typography Configurations */
            h1 {{
                font-size: 1.8rem !important; /* Shrunk from default massive sizes */
                margin-top: 0px !important;
            }}
            
            [data-testid="stMetricLabel"] {{
                font-size: 0.8rem !important; /* Sleeker labels */
                color: #94a3b8 !important;
                font-weight: 500;
            }}
            [data-testid="stMetricValue"] {{
                font-size: 1.35rem !important; /* Clean, compact values */
                font-weight: bold !important;
                color: #f8fafc !important;
            }}
            
            hr {{
                margin: 10px 0px !important;  /* Reduced divider margins */
                border-color: rgba(255, 255, 255, 0.15) !important;
            }}
            
            /* Shrink standard text spacing */
            .stMarkdown p {{
                font-size: 0.9rem !important;
                line-height: 1.4 !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning(f"Background asset missing. Please ensure your image is saved at: {image_path}")

# Run background injector
set_cluster_background("assets/cluster_bg.jpg")


# 3. Premium Brand Header Layout Integration
logo_col, title_col = st.columns([1, 10]) # Widened title column for a tighter row fit

###with logo_col:
    # Official Porsche Crest Vector (Slightly smaller for layout scaling)
    ###st.image(
      ###  "https://upload.wikimedia.org/wikipedia/en/d/df/Porsche_Crest.svg", 
       ## width=65
    ##)

with title_col:
    st.markdown("<h1 style='margin-bottom: 0px; padding-top: 0px;'>EcoPulse-AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.95rem; color: #cbd5e1 !important; margin-top: 0px;'>EcoPulse-AI: Machine Learning-Based Virtual Sensor</p>", unsafe_allow_html=True)

st.markdown("---")


# 4. Telemetry Metric Readouts Layout
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="CO₂ Prediction (R²)", value="0.982")

with col2:
    st.metric(label="NOx Prediction (R²)", value="0.982")

with col3:
    st.metric(label="CO₂ Error (RMSE)", value="0.177 g/h")

with col4:
    st.metric(label="NOx Error (RMSE)", value="0.127 g/h")

st.markdown("---")


# 5. Core Operational Description Card (Wrapped in a column block to look smaller)
desc_col, _ = st.columns([2, 1])
with desc_col:
    st.write(
        """
        **EcoPulse-AI** functions as an AI-powered virtual telemetry sensor that maps complex, 
        non-linear emissions directly from native engine operating parameters. By deploying trained deep 
        temporal networks (LSTM) and gradient-boosted trees (XGBoost), this software application eliminates 
        the operational requirement for physical hardware exhaust extraction probes.
        """
    )