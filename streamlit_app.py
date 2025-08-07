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

# 2. 🔧 Clean up column names (IMPORTANT!)
df.columns = df.columns.str.strip()

# 3. Show dataset columns (for debugging)
st.subheader("Dataset Columns (cleaned)")
st.write(df.columns.tolist())

# 4. Set target column
target_col = "Other_Faults"  # Make sure this matches cleaned name!

# 5. Display sample data
with st.expander("View raw dataset"):
    st.dataframe(df.head())

# 6. Split data
X = df.drop(target_col, axis=1)
y = df[target_col]

# 7. Input sliders
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
input_df = pd.DataFrame([input_data])

with st.expander("Your Input"):
    st.write(input_df)

# 8. Train model
model = RandomForestClassifier()
model.fit(X, y)

# 9. Predict
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# 10. Display result
st.subheader("Predicted Fault")
st.write(f"**{prediction[0]}**")

proba_df = pd.DataFrame(
    prediction_proba,
    columns=[f'Class {c}' for c in model.classes_]
)

st.subheader("Prediction Probabilities")
st.dataframe(proba_df.style.highlight_max(axis=1))

