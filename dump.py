import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="GeoVision  Rockfall Prediction", layout="wide")
st.title("GeoVision  Rockfall Risk Prediction")
st.caption("Trained ML-model for rockfall risk assessment")

# Load trained model
model = joblib.load("rockfall_model.pkl")

# Input: CSV or demo data
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
use_sample = st.button("Use demo data")



if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif use_sample:
    df = pd.read_csv("demo_data.csv")
else:
    df = None

if df is not None:
    features = ["slope_angle","rainfall_mm","vibration","pore_pressure","temperature_c"]
    X = df[features]

    # Predict risk levels
    df["predicted_risk_level"] = model.predict(X)

    # High risk subset, sorted
    high_risk = df[df["predicted_risk_level"]=="High"]
    high_risk = high_risk.sort_values(by=["slope_angle","rainfall_mm"], ascending=False)

    # Alert message
    if len(high_risk) > 0:
        st.warning(f"⚠️ {len(high_risk)} slopes detected as HIGH RISK!")

    # Styling: predicted_risk_level column
    def color_risk(val):
        if val == "Low": return "background-color: #22C55E; color: black"
        elif val == "Medium": return "background-color: #EAB308; color: black"
        else: return "background-color: #EF4444; color: black"

    styled_df = df.style.applymap(color_risk, subset=["predicted_risk_level"])

    # Display Main Table
    st.subheader("Prediction Results")
    st.dataframe(styled_df, use_container_width=True)

    # Display High Risk Table
    if len(high_risk) > 0:
        st.subheader("⚠️ High Risk Slopes")
        st.dataframe(high_risk, use_container_width=True)

    # Summary Metrics
    st.subheader("Summary Metrics")
    col1, col2 = st.columns(2)
    col1.metric("Total Slopes", len(df))
    col2.metric("High Risk Slopes", len(high_risk))

    # Charts side by side
    st.subheader("Charts")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Predicted Risk Distribution")
        fig, ax = plt.subplots(figsize=(4,3))
        risk_counts = df["predicted_risk_level"].value_counts()
        ax.bar(risk_counts.index, risk_counts.values, color=["#22C55E","#EAB308","#EF4444"])
        ax.set_ylabel("Number of Slopes")
        st.pyplot(fig)

    with col2:
        st.subheader("Feature Importance")
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            fig2, ax2 = plt.subplots(figsize=(4,3))
            ax2.barh(features, importances, color="skyblue")
            ax2.set_xlabel("Importance")
            st.pyplot(fig2)

    # Download Predictions
    st.download_button(
        "Download Predictions CSV",
        df.to_csv(index=False),
        "rockfall_predictions.csv",
        "text/csv"
    )

else:
    st.info("Upload a CSV or click **Use demo data** to get started.")
