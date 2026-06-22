import streamlit as st
import pandas as pd

st.title("📈 Model Performance")

results = pd.DataFrame({

    "Model":[
        "Linear Regression",
        "Random Forest",
        "XGBoost",
        "FNN",
        "LSTM"
    ],

    "Status":[
        "Completed",
        "Completed",
        "Completed",
        "Completed",
        "Best Model"
    ]
})

st.dataframe(results)