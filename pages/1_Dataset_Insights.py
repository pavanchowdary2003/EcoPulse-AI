import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.title("📊 Dataset Insights")

df = pd.read_csv("data/engine1.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Dataset Shape")
st.write(df.shape)

st.subheader("Summary Statistics")
st.dataframe(df.describe())

st.subheader("Correlation Heatmap")

# 1. Customizing figure with your larger structural canvas dimensions
fig, ax = plt.subplots(figsize=(14, 10))

# 2. Safety Check: Filter out non-numeric columns to avoid crashes
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()

# 3. Draw your stylized heatmap
sns.heatmap(
    corr_matrix,
    cmap="viridis",             # Uses your requested color palette
    annot=True,                 # Show correlation numbers inside cells
    fmt=".2f",                  # Round to two decimal places
    linewidths=0.5,             # Add elegant white borders around blocks
    linecolor="white",
    square=True,                # Keep forced 1:1 aspect ratio blocks
    cbar_kws={
        "shrink": 0.8, 
        "label": "Correlation Strength"
    },
    ax=ax
)

# 4. Integrate your modern typography configuration properties
ax.set_title(
    "Correlation Matrix – Engine1",
    fontsize=22,
    fontweight="bold",
    pad=20
)

# Apply specific custom label text rotations
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, rotation=45, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, rotation=0)

plt.tight_layout()

# 5. Render directly inside your Streamlit UI viewport
st.pyplot(fig)