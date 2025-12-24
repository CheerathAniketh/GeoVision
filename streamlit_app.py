import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="GeoVision – Rockfall Prediction", layout="wide")
st.title("GeoVision – Rockfall Risk Prediction")
st.caption("AI-powered rockfall risk assessment")

# Load trained model
model = joblib.load("rockfall_model.pkl")

# Input: CSV or demo data
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
use_sample = st.button("Use demo data")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif use_sample:
    df = pd.read_csv("synthetic_rockfall_1000.csv")
else:
    df = None

if df is not None:
    features = ["slope_angle","rainfall_mm","vibration","pore_pressure","temperature_c"]
    X = df[features]

    # Predict risk levels
    df["predicted_risk_level"] = model.predict(X)

    # High-risk probability
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        high_index = list(model.classes_).index("High")
        df["High_risk_prob"] = probs[:, high_index]

    # Highlight extreme slope+rainfall combo
    df["Critical"] = np.where((df["slope_angle"]>60) & (df["rainfall_mm"]>30), "Yes", "No")

    # Styling
    def color_risk(val):
        if val == "Low": return "background-color: #22C55E; color: black"
        elif val == "Medium": return "background-color: #EAB308; color: black"
        else: return "background-color: #EF4444; color: black"

    styled_df = df.style.applymap(color_risk, subset=["predicted_risk_level"])

    # Display Table
    st.subheader("📊 Prediction Results")
    st.dataframe(styled_df, use_container_width=True)

    # Summary Metrics
    st.subheader("Summary Metrics")
    high_risk = df[df["predicted_risk_level"]=="High"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Slopes", len(df))
    col2.metric("High Risk Slopes", len(high_risk))
    col3.metric("Max High-Risk Probability", round(df["High_risk_prob"].max(),2) if "High_risk_prob" in df.columns else "-")

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
    st.info("⬆️ Upload a CSV or click **Use demo data** to get started.")
