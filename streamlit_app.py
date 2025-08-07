import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('🌍 Earthquake Risk Detector')
st.write("This app predicts steel plate fault types to help assess potential earthquake vulnerabilities.")

# 1. Load data from GitHub
data_url = (
    "https://raw.githubusercontent.com/malekbenhafed/"
    "final-project/master/Steel_Plates_Faults.csv"
)
df = pd.read_csv(data_url)

# 2. Show dataset columns (for debugging)
st.subheader("Dataset Columns (use this list to identify target column)")
st.write(df.columns.tolist())

# 3. Specify correct target column name below
target_col = "fault 0"  # ← REPLACE this with the exact name from the columns list

# 4. Display sample data
with st.expander("View raw dataset"):
    st.dataframe(df.head())

# 5. Prepare features and target
X = df.drop(target_col, axis=1)
y = df[target_col]

# 6. Sidebar: input sliders for each feature
st.sidebar.header("Input Features")
input_data = {
    col: st.sidebar.slider(
        label=col,
        min_value=float(X[col].min()),
        max_value=float(X[col].max()),
        value=float(X[col].mean())
    )
    for col in X.columns
}

# Convert inputs to DataFrame
input_df = pd.DataFrame([input_data])

with st.expander("Your Input"):
    st.write(input_df)

# 7. Train Random Forest model
model = RandomForestClassifier()
model.fit(X, y)

# 8. Predict and show results
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("Predicted Fault Type")
st.write(f"**{prediction[0]}**")

proba_df = pd.DataFrame(
    prediction_proba,
    columns=[f'Class {c}' for c in model.classes_]
)

st.subheader("Prediction Probabilities")
st.dataframe(proba_df.style.highlight_max(axis=1))
