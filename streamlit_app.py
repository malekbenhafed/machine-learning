import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title('🌋 Earthquake Risk Detector')
st.info('This app predicts steel plate faults to assess earthquake risks!')

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('https://raw.githubusercontent.com/malekbenhafed/final-project/master/Steel_Plates_Faults.csv')

df = load_data()

# Debug: Show all columns
st.write("All columns in dataset:", list(df.columns))

# Define expected fault columns
expected_faults = ['Pastry', 'Z_Scratch', 'K_Scratch', 'Stains', 
                  'Dirtiness', 'Bumps', 'Other_Faults']

# Find which fault columns actually exist
existing_faults = [col for col in expected_faults if col in df.columns]
st.write("Existing fault columns:", existing_faults)

with st.expander('Data'):
    st.write('**Raw data (first 100 rows)**')
    st.dataframe(df.head(100))
    
    st.write('**X (Features)**')
    X_raw = df.drop(existing_faults, axis=1, errors='ignore')
    st.dataframe(X_raw)
    
    st.write('**y (Fault Types)**')
    if existing_faults:
        y_raw = df[existing_faults]
        st.dataframe(y_raw)
    else:
        st.error("No fault columns found in dataset!")

# Only proceed if we found fault columns
if existing_faults:
    # Use first existing fault as target
    target = existing_faults[0]
    
    with st.expander('Data visualization'):
        st.scatter_chart(data=df, x='X_Minimum', y='Y_Minimum', color=target)

    # Input features
    with st.sidebar:
        st.header('Steel Plate Parameters')
        
        # Check if steel type columns exist
        if 'TypeOfSteel_A300' in df.columns and 'TypeOfSteel_A400' in df.columns:
            steel_type = st.radio('Steel Type', ['A300', 'A400'])
        else:
            st.warning("Steel type columns not found")
            steel_type = 'A300'
        
        # Create sliders only for columns that exist
        num_features = {}
        for col in ['Steel_Plate_Thickness', 'Sum_of_Luminosity', 'Edges_Index']:
            if col in df.columns:
                num_features[col] = st.slider(
                    col,
                    float(df[col].min()),
                    float(df[col].max()),
                    float(df[col].mean())
                )
        
        # Create input DataFrame
        input_data = {}
        if 'TypeOfSteel_A300' in df.columns:
            input_data['TypeOfSteel_A300'] = [1 if steel_type == 'A300' else 0]
        if 'TypeOfSteel_A400' in df.columns:
            input_data['TypeOfSteel_A400'] = [1 if steel_type == 'A400' else 0]
        
        for col, val in num_features.items():
            input_data[col] = [val]
            
        input_df = pd.DataFrame(input_data)
        
    # Prepare data for model
    try:
        X = df.drop(existing_faults, axis=1, errors='ignore')
        y = df[target]
        
        # Model training
        clf = RandomForestClassifier()
        clf.fit(X, y)
        
        # Prediction
        prediction = clf.predict(input_df)
        prediction_proba = clf.predict_proba(input_df)
        
        # Display results
        st.subheader('Earthquake Risk Assessment')
        st.write(f'Probability of {target}:')
        
        df_prediction_proba = pd.DataFrame(
            prediction_proba, 
            columns=[f'No {target}', f'{target} Detected']
        )
        st.dataframe(df_prediction_proba.style.highlight_max(axis=1))
        
        if prediction[0] == 1:
            st.error('🚨 High earthquake risk detected!')
        else:
            st.success('✅ No significant risk detected')
            
    except Exception as e:
        st.error(f"Model error: {str(e)}")
else:
    st.error("Cannot proceed - no fault columns found in dataset")
