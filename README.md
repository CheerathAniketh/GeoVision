# ML-Based Rockfall Prediction System

A machine learning web app that predicts potential rockfall risk in open-pit mining environments, helping improve safety and preparedness.

---

## Problem Statement

Rockfalls are a major safety and operational risk in open-pit mines. Early prediction of high-risk conditions can help prevent accidents and reduce damage to personnel and equipment.

---

## Solution

This project uses a **Random Forest Regression** model to analyze multiple input parameters and predict rockfall risk.  
Predictions are displayed through an **interactive Streamlit dashboard** for quick and easy decision support.

**Check it out live:** [GeoVision360](https://geovision360.streamlit.app/)

---

## Key Features

- AI-powered rockfall risk prediction  
- Real-time user input and instant results  
- Interactive Streamlit dashboard  
- End-to-end ML pipeline (training → deployment)  

---

## Tech Stack

- Python  
- Streamlit  
- Pandas & NumPy  
- Scikit-learn  
- Matplotlib  
- Joblib  

---

## Project Structure

```
├── streamlit_app.py           # Streamlit application
├── rockfall_model.pkl         # Trained ML model
├── requirements.txt           # Project dependencies
└── data/                      # Input dataset
```

---

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## Impact

- Enhances situational awareness in mining operations
- Supports proactive safety measures
- Demonstrates practical AI application in industrial safety

---

## Disclaimer

This project is a hackathon prototype and should not be used for real-world safety decisions.
