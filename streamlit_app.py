import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Configuration
st.set_page_config(
    page_title="GeoVision Rockfall Prediction",
    layout="wide"
)

st.title("GeoVision – Rockfall Risk Prediction")
st.caption("ML-powered risk assessment for open-pit mining")

# Constants
REQUIRED_COLS = [
    "slope_angle",
    "rainfall_mm",
    "vibration",
    "pore_pressure",
    "temperature_c"
]

# Model loading
model = joblib.load("rockfall_model.pkl")

# User Input
st.info(
    "**CSV format required**:\n"
    "slope_angle, rainfall_mm, vibration, pore_pressure, temperature_c\n\n"
)

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
use_sample = st.button("Use demo data")

# Loading Data
df = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing_cols:
        st.error(
            "Invalid CSV format\n\n"
            f"Missing columns: {', '.join(missing_cols)}\n\n"
            "Expected columns:\n"
            "slope_angle, rainfall_mm, vibration, pore_pressure, temperature_c"
        )
        st.stop()

elif use_sample:
    df = pd.read_csv("demo_data.csv")

# Core Logic
if df is not None:

    X = df[REQUIRED_COLS]
    df["predicted_risk_level"] = model.predict(X)

    high_risk = (
        df[df["predicted_risk_level"] == "High"]
        .sort_values(by=["slope_angle", "rainfall_mm"], ascending=False)
    )

    if len(high_risk) > 0:
        st.warning(f"⚠️ {len(high_risk)} slopes detected as **HIGH RISK**")

    # Table Styling
    def color_risk(val):
        if val == "Low":
            return "background-color: #22C55E; color: black"
        elif val == "Medium":
            return "background-color: #EAB308; color: black"
        else:
            return "background-color: #EF4444; color: black"

    styled_df = df.style.applymap(
        color_risk, subset=["predicted_risk_level"]
    )

    st.subheader("Prediction Results")
    st.dataframe(styled_df, use_container_width=True)

    if len(high_risk) > 0:
        st.subheader("⚠️ High Risk Slopes")
        st.dataframe(high_risk, use_container_width=True)

    # Metrics
    st.subheader("Summary Metrics")
    col1, col2 = st.columns(2)
    col1.metric("Total Slopes", len(df))
    col2.metric("High Risk Slopes", len(high_risk))

    # Charts
    st.subheader("Visual Insights")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Distribution")
        fig, ax = plt.subplots(figsize=(4, 3))
        risk_counts = df["predicted_risk_level"].value_counts()
        ax.bar(
            risk_counts.index,
            risk_counts.values,
            color=["#22C55E", "#EAB308", "#EF4444"]
        )
        ax.set_ylabel("Number of Slopes")
        st.pyplot(fig)

    with col2:
        st.subheader("Feature Importance")
        if hasattr(model, "feature_importances_"):
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            ax2.barh(REQUIRED_COLS, model.feature_importances_)
            ax2.set_xlabel("Importance")
            st.pyplot(fig2)
        else:
            st.info("Feature importance not supported by this model.")

    # Download button
    st.download_button(
        "Download Predictions (CSV)",
        df.to_csv(index=False),
        "rockfall_predictions.csv",
        "text/csv"
    )

else:
    st.info("Upload a CSV file or click **Use demo data** to begin ")
