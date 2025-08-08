import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# App title and description
st.title('🌋 Earthquake Risk Detector')
st.info('This app predicts steel plate faults to assess earthquake risks!')

# Load data
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/malekbenhafed/final-project/master/Steel_Plates_Faults.csv'
    df = pd.read_csv(url)
    
    # Clean column names (remove extra spaces if any)
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# Debug: Show all columns
st.write("All columns in dataset:", list(df.columns))

# Define possible fault columns (all possible names)
possible_faults = ['Pastry', 'Z_Scratch', 'K_Scratch', 'Stains', 
                  'Dirtiness', 'Bumps', 'Other_Faults']

# Find which fault columns actually exist
fault_columns = [col for col in possible_faults if col in df.columns]

if not fault_columns:
    st.error("CRITICAL ERROR: No fault columns found in dataset!")
    st.stop()

# Data display
with st.expander('Data'):
    st.write('**Raw data (first 100 rows)**')
    st.dataframe(df.head(100))
    
    st.write('**X (Features)**')
    X_raw = df.drop(fault_columns, axis=1, errors='ignore')
    st.dataframe(X_raw)
    
    st.write('**y (Fault Types)**')
    y_raw = df[fault_columns]
    st.dataframe(y_raw)

# Visualization
with st.expander('Data visualization'):
    if 'X_Minimum' in df.columns and 'Y_Minimum' in df.columns:
        st.scatter_chart(data=df.head(500), x='X_Minimum', y='Y_Minimum', 
                        color=fault_columns[0] if fault_columns else None)
    else:
        st.warning("Cannot create visualization - missing coordinate columns")

# Input features in sidebar
with st.sidebar:
    st.header('Steel Plate Parameters')
    
    # Steel type selection
    if 'TypeOfSteel_A300' in df.columns and 'TypeOfSteel_A400' in df.columns:
        steel_type = st.radio('Steel Type', ['A300', 'A400'])
    else:
        steel_type = 'A300'
        st.warning("Steel type columns not found - using default")
    
    # Numeric features - create sliders only for columns that exist
    numeric_features = {}
    for col in ['Steel_Plate_Thickness', 'Sum_of_Luminosity', 'Edges_Index',
               'X_Minimum', 'Y_Minimum', 'Empty_Index', 'LogOfAreas']:
        if col in df.columns:
            numeric_features[col] = st.slider(
                col.replace('_', ' '),
                float(df[col].min()),
                float(df[col].max()),
                float(df[col].mean())
            )
    
    # Create input DataFrame with all required features
    input_data = {}
    
    # Add steel type
    if 'TypeOfSteel_A300' in df.columns:
        input_data['TypeOfSteel_A300'] = [1 if steel_type == 'A300' else 0]
    if 'TypeOfSteel_A400' in df.columns:
        input_data['TypeOfSteel_A400'] = [1 if steel_type == 'A400' else 0]
    
    # Add numeric features
    for col, val in numeric_features.items():
        input_data[col] = [val]
    
    # Fill missing columns with mean values
    for col in X_raw.columns:
        if col not in input_data:
            input_data[col] = [X_raw[col].mean()]

# Prepare data for modeling
try:
    X = df.drop(fault_columns, axis=1, errors='ignore')
    y = df[fault_columns[0]]  # Using first fault type as target
    
    # Train model
    clf = RandomForestClassifier()
    clf.fit(X, y)
    
    # Prepare input for prediction
    input_df = pd.DataFrame(input_data)
    
    # Ensure columns match training data exactly
    input_df = input_df[X.columns]
    
    # Make prediction
    prediction = clf.predict(input_df)
    prediction_proba = clf.predict_proba(input_df)
    
    # Display results
    st.subheader('Earthquake Risk Assessment')
    st.write(f'Probability of {fault_columns[0]}:')
    
    df_prediction_proba = pd.DataFrame(
        prediction_proba, 
        columns=[f'No {fault_columns[0]}', f'{fault_columns[0]} Detected']
    )
    st.dataframe(df_prediction_proba.style.highlight_max(axis=1))
    
    if prediction[0] == 1:
        st.error('🚨 High earthquake risk detected!')
    else:
        st.success('✅ No significant risk detected')

except Exception as e:
    st.error(f"Model error: {str(e)}")
    st.write("Please check if all required features are available in your data")
