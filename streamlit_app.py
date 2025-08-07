import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title('🌍 Earthquake Risk Detector')
st.write("This app uses machine learning to detect possible steel plate faults which might indicate earthquake-related risks.")

# Load data
df = pd.read_csv('https://raw.githubusercontent.com/malekbenhafed/final-project/master/Steel_Plates_Faults.csv')

# Show columns to identify target
st.write("### 📊 Dataset Columns:")
st.write(df.columns)

# Identify target column
target_col = 'Faults'  # Use this if you know the correct target column (change if needed)

# Display raw data
with st.expander('🔍 View raw dataset'):
    st.write(df)

# Prepare data
X = df.drop(target_col, axis=1)
y = df[target_col]

# Sidebar inputs for features
st.sidebar.header('📥 Input Features')

# Take one row from X to get the feature names and example ranges
sample = X.iloc[0]

# Create input sliders dynamically
input_data = {}
for col in X.columns:
    min_val = float(X[col].min())
    max_val = float(X[col].max())
    default_val = float(X[col].mean())
    input_data[col] = st.sidebar.slider(f"{col}", min_val, max_val, default_val)

# Convert inputs to DataFrame
input_df = pd.DataFrame([input_data])

# Display input data
with st.expander("📌 Input data"):
    st.write(input_df)

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Predict
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Show results
st.subheader("🧠 Prediction")
st.success(f"Predicted Fault Type: {int(prediction[0])}")

st.subheader("📈 Prediction Probabilities")
proba_df = pd.DataFrame(prediction_proba, co
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
