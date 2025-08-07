import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Title
st.title("🌍 Earthquake Risk Detector")

st.write("This app predicts possible fault types in steel plates, which may help anticipate structural risks before an earthquake.")

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/malekbenhafed/final-project/master/Steel_Plates_Faults.csv')

# Show raw data
with st.expander("Raw Data"):
    st.dataframe(df)

# Prepare features (X) and target (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Sidebar for user input
st.sidebar.header("Input Features")

# Auto-generate sliders for numeric input
input_data = {}
for col in X.columns:
    min_val = float(X[col].min())
    max_val = float(X[col].max())
    mean_val = float(X[col].mean())
    input_data[col] = st.sidebar.slider(
        col, min_value=min_val, max_value=max_val, value=mean_val
    )

# Create DataFrame for input
input_df = pd.DataFrame([input_data])

# Show input features
with st.expander("Input Features Preview"):
    st.write(input_df)

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Make prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Show prediction
st.subheader("Predicted Fault Class")
st.write(f"Prediction: **{prediction[0]}**")

# Show probability
st.subheader("Prediction Probabilities")
proba_df = pd.DataFrame(prediction_proba, columns=[f'Class {c}' for c in model.classes_])
st.dataframe(proba_df.style.highlight_max(axis=1))
