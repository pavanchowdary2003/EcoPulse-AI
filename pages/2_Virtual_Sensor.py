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
            div[data-testid="stSlider"], [data-testid="stMetric"], div[data-testid="stVerticalBlock"] > div, .stButton, div[data-baseweb="select"] {{
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

# Run background injector
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

# Helper function to map human values to model space [-5, 5]
def scale_to_model(value, real_min, real_max):
    return -5.0 + 10.0 * (value - real_min) / (real_max - real_min)

try:
    model = load_lstm_core()

    # 5. Real-world Input Sliders divided into multi-column layout split rows
    col_left, col_right = st.columns(2)
    
    with col_left:
        speed_rpm = st.slider("Engine Speed", min_value=0, max_value=350, value=175, step=5)
        load_pct = st.slider("Engine Load (%)", min_value=10, max_value=100, value=50, step=1)
        lambda_val = st.slider("Lambda (λ)", min_value=0.80, max_value=1.40, value=1.00, step=0.01)

    with col_right:
        ignition_deg = st.slider("Ignition Angle (°BTDC)", min_value=0, max_value=40, value=15, step=1)
        fuel_cutoff = st.selectbox("Fuel Cutoff State", options=[0, 1], format_func=lambda x: "Active (1)" if x == 1 else "Inactive (0)")

    # 6. Transform inputs behind the scenes into model space [-5, 5]
    speed = scale_to_model(speed_rpm, 0.0, 350.0)
    load = scale_to_model(load_pct, 10.0, 100.0)
    lambda_ = scale_to_model(lambda_val, 0.80, 1.40)
    ignition = scale_to_model(ignition_deg, 0.0, 40.0)
    fuel = 5.0 if fuel_cutoff == 1 else -5.0
    
    # Computed normalized features calculated dynamically
    speed_load = speed * load
    lambda_load = lambda_ * load
    temp_diff = 0.0

    st.markdown("---")

    # 7. Prediction Activation Trigger and Metric Readouts
    if st.button("Predict Emissions Output", use_container_width=True):
        # Package 8 features expected by model configuration
        x = np.array([[
            speed, load, lambda_, ignition, fuel,
            speed_load, lambda_load, temp_diff
        ]], dtype=np.float32)

        # Structure shape translation matching LSTM requirements (1, 20, 8)
        x = np.repeat(x.reshape(1, 1, 8), 20, axis=1)

        # Generate standard-scaled predictions
        pred = model.predict(x)
        raw_co2_pred = float(pred[0][0])
        raw_nox_pred = float(pred[0][1])

        # 8. INVERSE SCALING PARAMETERS (De-standardization)
        # TODO: Replace these placeholders with the Mean (μ) and Std Dev (σ) from your training set scaling!
        co2_mean, co2_std = 2100.0, 450.0   
        nox_mean, nox_std = 35.0, 12.0      
        
        # Reverse Standardization Math: Real = (Scaled * Std) + Mean
        real_co2 = (raw_co2_pred * co2_std) + co2_mean
        real_nox = (raw_nox_pred * nox_std) + nox_mean

        # Side-by-side metric layout rendering
        out1, out2 = st.columns(2)
        with out1:
            st.metric("Predicted CO₂ Emissions", f"{real_co2:.2f} g/h")
        with out2:
            st.metric("Predicted NOx Emissions", f"{real_nox:.2f} g/h")

except Exception as e:
    st.error(f"Inference model offline. Error details: {e}")