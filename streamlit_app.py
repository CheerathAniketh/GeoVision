import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="GeoVision – Rockfall Prediction", layout="wide")
st.title("GeoVision – Rockfall Risk Prediction")
st.caption("AI-powered rockfall risk assessment")

# Load new model
model = joblib.load("rockfall_model.pkl")

# Input
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
    df["predicted_risk_level"] = model.predict(X)

    # Styling
    def color_risk(val):
        if val == "Low": return "background-color: #22C55E; color: black"
        elif val == "Medium": return "background-color: #EAB308; color: black"
        else: return "background-color: #EF4444; color: black"

    styled_df = df.style.applymap(color_risk, subset=["predicted_risk_level"])

    # Output
    st.subheader("📊 Prediction Results")
    st.dataframe(styled_df, use_container_width=True)

    # Alerts
    high_risk = df[df["predicted_risk_level"] == "High"]
    if not high_risk.empty:
        st.warning("🚨 High-risk slopes detected!")
        st.dataframe(high_risk, use_container_width=True)

    # Summary
    st.subheader("Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Slopes", len(df))
    col2.metric("High Risk Slopes", len(high_risk))
    col3.metric("Risk Levels", ", ".join(df["predicted_risk_level"].value_counts().index))

else:
    st.info("⬆️ Upload a CSV or click **Use demo data** to get started.")
