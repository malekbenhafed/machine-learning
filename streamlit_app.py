import streamlit as st
import pandas as pd

st.title('🌍 Earthquake Risk Detector')
st.write("This app predicts steel plate fault types to help assess potential earthquake vulnerabilities.")

with st.expander('Data'):
    st.write('**Raw data**')
    df = pd.read_csv('https://raw.githubusercontent.com/malekbenhafed/final-project/master/Steel_Plates_Faults.csv')
    
    # First display the raw data
    st.dataframe(df)
    
    # Show all column names to debug
    st.write("**Column names in dataset:**")
    st.write(list(df.columns))
    
    # Define the fault columns we expect
    expected_fault_columns = ['Pastry', 'Z_Scratch', 'K_Scratch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
    
    # Find which fault columns actually exist in the data
    existing_fault_columns = [col for col in expected_fault_columns if col in df.columns]
    
    st.write('**X (Features)**')
    X_raw = df.drop(existing_fault_columns, axis=1)
with st.expander('Data visualization'):
    st.scatter_chart(data=df, x='X_Minimum', y='Y_Minimum', color='Pastry')
    st.dataframe(X_raw)

    st.write('**y (Target - Fault Types)**')
    y_raw = df[existing_fault_columns]
    st.dataframe(y_raw)
    # Input features
with st.sidebar:
    st.header('Steel Plate Features')
    steel_type = st.selectbox('Type of Steel', ('A300', 'A400'))
    thickness = st.slider('Plate Thickness', float(df['Steel_Plate_Thickness'].min()), 
                         float(df['Steel_Plate_Thickness'].max()), 
                         float(df['Steel_Plate_Thickness'].median()))
    luminosity = st.slider('Sum of Luminosity', float(df['Sum_of_Luminosity'].min()),
                          float(df['Sum_of_Luminosity'].max()),
                          float(df['Sum_of_Luminosity'].median()))
    x_position = st.slider('X Position', float(df['X_Minimum'].min()),
                          float(df['X_Minimum'].max()),
                          float(df['X_Minimum'].median()))
