import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import os

# 1. Initialize Page Geometry Configuration
st.set_page_config(
    page_title="Model Explainability",
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
            div[data-testid="stSelectbox"], div[data-testid="stVerticalBlock"] > div, .stTabs {{
                background-color: rgba(15, 23, 42, 0.75) !important;
                border-radius: 8px;
                padding: 12px 16px !important;
                margin-bottom: 8px !important;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
                backdrop-filter: blur(6px);
                -webkit-backdrop-filter: blur(6px);
                border: 1px solid rgba(255, 255, 255, 0.12);
            }}
            
            /* Text Color Alignment */
            h1, h2, h3, p, span, label, div {{
                color: #ffffff !important;
            }}
            
            h1 {{
                font-size: 1.8rem !important;
                margin-top: 0px !important;
            }}
            h3 {{
                font-size: 1.2rem !important;
                font-weight: 600 !important;
                margin-bottom: 10px !important;
            }}
            
            button[data-baseweb="tab"] {{
                color: #94a3b8 !important;
            }}
            button[aria-selected="true"] {{
                color: #ffffff !important;
                font-weight: bold !important;
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
    st.markdown("<h1 style='margin-bottom: 0px; padding-top: 0px;'>Model Explainability Suite</h1>", unsafe_allow_html=True)

st.markdown("---")

# 4. Fast Cache Model Loading
@st.cache_resource
def load_xgb_core():
    return joblib.load("models/xgb_model.pkl")

try:
    xgb = load_xgb_core()
    features = ["Speed", "Load", "Lambda", "Ignition_Angle", "Fuel_Cutoff", "Speed_Load", "Lambda_Load", "Temp_Diff"]
    
    # --- FIX: Safe Multi-Output Extraction ---
    # MultiOutputRegressor wraps individual estimators inside .estimators_
    if hasattr(xgb, "estimators_"):
        co2_importance = xgb.estimators_[0].feature_importances_
        nox_importance = xgb.estimators_[1].feature_importances_
    else:
        co2_importance = xgb.feature_importances_
        nox_importance = xgb.feature_importances_

    # Create distinct sorted dataframes for each target
    df_co2 = pd.DataFrame({"Feature": features, "Importance": co2_importance}).sort_values(by="Importance", ascending=True)
    df_nox = pd.DataFrame({"Feature": features, "Importance": nox_importance}).sort_values(by="Importance", ascending=True)

    # 5. Create Interactive Analytical Tabs
    tab1, tab2 = st.tabs(["📊 Global Feature Importance", "📈 Parameter Sensitivity Profiler"])

    with tab1:
        st.subheader("Global Feature Importance Breakdown")
        st.write("Compare which parameters drive individual $CO_2$ changes versus $NO_x$ changes inside the model ensemble.")
        
        # Split screen horizontally into 2 clean columns
        col_co2, col_nox = st.columns(2)
        
        with col_co2:
            st.markdown("##### **$CO_2$ Model Drivers**")
            fig1, ax1 = plt.subplots(figsize=(6, 4.5), dpi=110)
            fig1.patch.set_facecolor('none')
            ax1.set_facecolor('none')
            
            sns.barplot(data=df_co2, x="Importance", y="Feature", palette="viridis", hue="Feature", legend=False, ax=ax1)
            ax1.bar_label(ax1.containers[0], fmt="%.2f", padding=5, color="#ffffff", fontsize=8)
            ax1.tick_params(colors='#ffffff', labelsize=8)
            ax1.set_xlabel("Importance", color="#cbd5e1", fontsize=8)
            ax1.set_ylabel("", color="#cbd5e1")
            sns.despine(left=True, bottom=True)
            ax1.grid(axis='x', linestyle='--', alpha=0.1)
            plt.tight_layout()
            st.pyplot(fig1)

        with col_nox:
            st.markdown("##### **$NO_x$ Model Drivers**")
            fig2, ax2 = plt.subplots(figsize=(6, 4.5), dpi=110)
            fig2.patch.set_facecolor('none')
            ax2.set_facecolor('none')
            
            sns.barplot(data=df_nox, x="Importance", y="Feature", palette="magma", hue="Feature", legend=False, ax=ax2)
            ax2.bar_label(ax2.containers[0], fmt="%.2f", padding=5, color="#ffffff", fontsize=8)
            ax2.tick_params(colors='#ffffff', labelsize=8)
            ax2.set_xlabel("Importance", color="#cbd5e1", fontsize=8)
            ax2.set_ylabel("", color="#cbd5e1")
            sns.despine(left=True, bottom=True)
            ax2.grid(axis='x', linestyle='--', alpha=0.1)
            plt.tight_layout()
            st.pyplot(fig2)

    with tab2:
        st.subheader("What-If Single Parameter Sensitivity Profiling")
        target_param = st.selectbox("Isolate Engine Parameter for Testing Target Curve:", features)
        
        sweep_axis = np.linspace(-5.0, 5.0, 100)
        base_input = np.zeros((100, len(features)))
        target_idx = features.index(target_param)
        base_input[:, target_idx] = sweep_axis
        
        if target_param in ["Speed", "Load"]:
            base_input[:, features.index("Speed_Load")] = base_input[:, features.index("Speed")] * base_input[:, features.index("Load")]
        if target_param in ["Lambda", "Load"]:
            base_input[:, features.index("Lambda_Load")] = base_input[:, features.index("Lambda")] * base_input[:, features.index("Load")]

        sensitivity_predictions = xgb.predict(base_input)

        col_curve1, col_curve2 = st.columns(2)
        
        with col_curve1:
            st.markdown("##### **$CO_2$ Dynamic Response Curve**")
            fig3, ax3 = plt.subplots(figsize=(6, 3.8), dpi=110)
            fig3.patch.set_facecolor('none')
            ax3.set_facecolor('none')
            ax3.plot(sweep_axis, sensitivity_predictions[:, 0], color="#38bdf8", linewidth=2.5)
            ax3.fill_between(sweep_axis, sensitivity_predictions[:, 0], sensitivity_predictions[:, 0].min(), color="#38bdf8", alpha=0.1)
            ax3.set_xlabel(f"{target_param} Range", color="#cbd5e1", fontsize=8)
            ax3.tick_params(colors='#ffffff', labelsize=8)
            sns.despine()
            ax3.grid(linestyle='--', alpha=0.1)
            plt.tight_layout()
            st.pyplot(fig3)

        with col_curve2:
            st.markdown("##### **$NO_x$ Dynamic Response Curve**")
            fig4, ax4 = plt.subplots(figsize=(6, 3.8), dpi=110)
            fig4.patch.set_facecolor('none')
            ax4.set_facecolor('none')
            ax4.plot(sweep_axis, sensitivity_predictions[:, 1], color="#f43f5e", linewidth=2.5)
            ax4.fill_between(sweep_axis, sensitivity_predictions[:, 1], sensitivity_predictions[:, 1].min(), color="#f43f5e", alpha=0.1)
            ax4.set_xlabel(f"{target_param} Range", color="#cbd5e1", fontsize=8)
            ax4.tick_params(colors='#ffffff', labelsize=8)
            sns.despine()
            ax4.grid(linestyle='--', alpha=0.1)
            plt.tight_layout()
            st.pyplot(fig4)

except Exception as e:
    st.error(f"Explainability suite locked out. Error details: {e}")