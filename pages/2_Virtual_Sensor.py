import streamlit as st
import numpy as np
import base64
import os
from tensorflow.keras.models import load_model

# 1. Initialize Page Geometry Configuration
st.set_page_config(
    page_title="Virtual Sensor",
    page_icon="⚡",
    layout="wide"
)

# 2. Automotive Instrument Cluster Background Engine
def set_cluster_background(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        
        # Inject structural layout CSS with smaller glass boxes
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
            div[data-testid="stSlider"], [data-testid="stMetric"], div[data-testid="stVerticalBlock"] > div, .stButton {{
                background-color: rgba(15, 23, 42, 0.75) !important;
                border-radius: 8px;
                padding: 10px 14px !important;
                margin-bottom: 6px !important;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
                backdrop-filter: blur(6px);
                -webkit-backdrop-filter: blur(6px);
                border: 1px solid rgba(255, 255, 255, 0.12);
            }}
            
            /* High-contrast text sizing */
            h1, h2, h3, p, span, label, div {{
                color: #ffffff !important;
            }}
            
            h1 {{
                font-size: 1.8rem !important;
                margin-top: 0px !important;
            }}
            
            [data-testid="stMetricLabel"] {{
                font-size: 0.8rem !important;
                color: #94a3b8 !important;
            }}
            
            [data-testid="stMetricValue"] {{
                font-size: 1.35rem !important;
                font-weight: bold !important;
            }}
            
            /* Tighten up space between elements */
            .stSlider [data-testid="stWidgetLabel"] p {{
                font-size: 0.85rem !important;
                font-weight: 600 !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning(f"Background image missing. Please save it as: {image_path}")

# Run background injector (Make sure your uploaded file is inside assets/ named unnamed.jpg)
set_cluster_background("assets/unnamed.jpg")


# 3. Premium Compact Title Banner
logo_col, title_col = st.columns([1, 12])
with logo_col:
    st.markdown("<h1 style='margin-top: 0px; padding-top: 0px; text-align: center;'>🚗</h1>", unsafe_allow_html=True)
with title_col:
    st.markdown("<h1 style='margin-bottom: 0px; padding-top: 0px;'>Virtual Sensor</h1>", unsafe_allow_html=True)

st.markdown("---")


# 4. Fast Cache Model Allocation Link
@st.cache_resource
def load_lstm_core():
    return load_model("models/lstm_model.keras")

try:
    model = load_lstm_core()

    # 5. Compact Input Sliders divided into multi-column layout split rows
    col_left, col_right = st.columns(2)
    
    with col_left:
        speed = st.slider("Speed", -5.0, 5.0, 0.0, step=0.1)
        load = st.slider("Load", -5.0, 5.0, 0.0, step=0.1)
        lambda_ = st.slider("Lambda", -5.0, 5.0, 0.0, step=0.1)

    with col_right:
        ignition = st.slider("Ignition Angle", -5.0, 5.0, 0.0, step=0.1)
        fuel = st.slider("Fuel Cutoff", -5.0, 5.0, 0.0, step=0.1)
        
        # Computed features calculated dynamically
        speed_load = speed * load
        lambda_load = lambda_ * load
        temp_diff = 0.0

    st.markdown("---")

    # 6. Prediction Activation Trigger and Metric Readouts
    if st.button("Predict Emissions Output", use_container_width=True):
        x = np.array([[
            speed, load, lambda_, ignition, fuel,
            speed_load, lambda_load, temp_diff
        ]], dtype=np.float32)

        # Structure shape translation matching requirements (1, 20, 8)
        x = np.repeat(x.reshape(1, 1, 8), 20, axis=1)

        pred = model.predict(x)

        # Side-by-side metric layout rendering
        out1, out2 = st.columns(2)
        with out1:
            st.metric("Predicted CO₂ Intensity", f"{round(float(pred[0][0]), 3)} g/h")
        with out2:
            st.metric("Predicted NOx Intensity", f"{round(float(pred[0][1]), 3)} g/h")

except Exception as e:
    st.error(f"Inference model offline. Error details: {e}")