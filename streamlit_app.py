import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title('🌋 Earthquake Risk Detector')
st.info('This app predicts steel plate faults to assess earthquake risks!')

# Load data
df = pd.read_csv('https://raw.githubusercontent.com/malekbenhafed/final-project/master/Steel_Plates_Faults.csv')

with st.expander('Data'):
    st.write('**Raw data**')
    df

    st.write('**X**')
    fault_columns = ['Pastry', 'Z_Scratch', 'K_Scratch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
    X_raw = df.drop(fault_columns, axis=1)
    X_raw

    st.write('**y**')
    y_raw = df[fault_columns]
    y_raw

with st.expander('Data visualization'):
    st.scatter_chart(data=df, x='X_Minimum', y='Y_Minimum', color='Other_Faults')

# Input features - MUST INCLUDE ALL COLUMNS USED IN TRAINING
with st.sidebar:
    st.header('Input features')
    
    # Steel type
    steel_type = st.selectbox('Steel Type', ['A300', 'A400'])
    
    # Critical features that were in the error message
    edges_x_index = st.slider('Edges_X_Index', 
                            float(df['Edges_X_Index'].min()),
                            float(df['Edges_X_Index'].max()),
                            float(df['Edges_X_Index'].mean()))
    
    edges_y_index = st.slider('Edges_Y_Index',
                            float(df['Edges_Y_Index'].min()),
                            float(df['Edges_Y_Index'].max()),
                            float(df['Edges_Y_Index'].mean()))
    
    empty_index = st.slider('Empty_Index',
                          float(df['Empty_Index'].min()),
                          float(df['Empty_Index'].max()),
                          float(df['Empty_Index'].mean()))
    
    log_areas = st.slider('LogOfAreas',
                        float(df['LogOfAreas'].min()),
                        float(df['LogOfAreas'].max()),
                        float(df['LogOfAreas'].mean()))
    
    # Add other required features here...
    
    # Create input DataFrame with ALL features used in training
    input_data = {
        'TypeOfSteel_A300': [1 if steel_type == 'A300' else 0],
        'TypeOfSteel_A400': [1 if steel_type == 'A400' else 0],
        'Edges_X_Index': [edges_x_index],
        'Edges_Y_Index': [edges_y_index],
        'Empty_Index': [empty_index],
        'LogOfAreas': [log_areas],
        # Add all other features exactly as they appear in X_raw
    }
    
    # Ensure we include all columns that were in the training data
    for col in X_raw.columns:
        if col not in input_data:
            input_data[col] = [X_raw[col].mean()]  # Fill with mean if not in input
    
    input_df = pd.DataFrame(input_data)
    input_combined = pd.concat([input_df, X_raw], axis=0)

# Data preparation
X = input_combined[1:]
input_row = input_combined[:1]

# Model training (using first fault type as example)
clf = RandomForestClassifier()
clf.fit(X, df['Other_Faults'])

# Prediction
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

# Display results
st.subheader('Earthquake Risk Assessment')
st.write('Probability of fault detection:')

df_prediction_proba = pd.DataFrame(prediction_proba, columns=['No Fault', 'Fault Detected'])
st.dataframe(df_prediction_proba.style.highlight_max(axis=1))

if prediction[0] == 1:
    st.error('🚨 High earthquake risk detected!')
else:
    st.success('✅ No significant risk detected')
