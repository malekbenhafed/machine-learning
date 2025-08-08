import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('🌋 Earthquake Risk Detector')
st.info('This app predicts steel plate faults to assess earthquake risks!')

# Load data
df = pd.read_csv('https://raw.githubusercontent.com/malekbenhafed/final-project/master/Steel_Plates_Faults.csv')

with st.expander('Data'):
    st.write('**Raw data**')
    st.dataframe(df)
    
    st.write('**X (Features)**')
    fault_columns = ['Pastry', 'Z_Scratch', 'K_Scratch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
    X_raw = df.drop(fault_columns, axis=1)
    st.dataframe(X_raw)
    
    st.write('**y (Fault Types)**')
    y_raw = df[fault_columns]
    st.dataframe(y_raw)

with st.expander('Data visualization'):
    st.scatter_chart(data=df, x='X_Minimum', y='Y_Minimum', color='Other_Faults')

# Input features
with st.sidebar:
    st.header('Steel Plate Parameters')
    
    steel_type = st.selectbox('Steel Type', ['A300', 'A400'])
    thickness = st.slider('Plate Thickness', 
                        float(df['Steel_Plate_Thickness'].min()),
                        float(df['Steel_Plate_Thickness'].max()),
                        float(df['Steel_Plate_Thickness'].mean()))
    
    luminosity = st.slider('Luminosity', 
                         float(df['Sum_of_Luminosity'].min()),
                         float(df['Sum_of_Luminosity'].max()),
                         float(df['Sum_of_Luminosity'].mean()))
    
    edge_index = st.slider('Edge Index',
                         float(df['Edges_Index'].min()),
                         float(df['Edges_Index'].max()),
                         float(df['Edges_Index'].mean()))
    
    # Create input DataFrame
    input_data = {
        'TypeOfSteel_A300': [1 if steel_type == 'A300' else 0],
        'TypeOfSteel_A400': [1 if steel_type == 'A400' else 0],
        'Steel_Plate_Thickness': [thickness],
        'Sum_of_Luminosity': [luminosity],
        'Edges_Index': [edge_index]
    }
    input_df = pd.DataFrame(input_data)
    input_combined = pd.concat([input_df, X_raw], axis=0)

# Data preparation
X = input_combined[1:]
input_row = input_combined[:1]

# Model training (using first fault type as example)
clf = RandomForestClassifier()
clf.fit(X, df['Other_Faults'])  # Using 'Other_Faults' as target

# Prediction
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

# Display results
st.subheader('Earthquake Risk Assessment')
st.write('Probability of fault detection:')

df_prediction_proba = pd.DataFrame(prediction_proba, columns=['No Fault', 'Fault Detected'])
st.dataframe(df_prediction_proba.style.highlight_max(axis=1), 
             use_container_width=True)

if prediction[0] == 1:
    st.error('🚨 High earthquake risk detected!')
else:
    st.success('✅ No significant risk detected')
