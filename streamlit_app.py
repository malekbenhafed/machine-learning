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
  # Earthquake Detection Input Features
with st.sidebar:
    st.header('🌋 Earthquake Risk Parameters')
    
    # Plate characteristics
    plate_type = st.selectbox('Steel Plate Type', ['A300', 'A400'])
    plate_thickness = st.slider('Plate Thickness (mm)', 
                              min_value=float(df['Steel_Plate_Thickness'].min()),
                              max_value=float(df['Steel_Plate_Thickness'].max()),
                              value=float(df['Steel_Plate_Thickness'].median()))
    
    # Structural features
    edge_index = st.slider('Edge Stress Index', 
                         min_value=float(df['Edges_Index'].min()),
                         max_value=float(df['Edges_Index'].max()),
                         value=float(df['Edges_Index'].mean()))
    
    # Seismic activity indicators
    luminosity = st.slider('Microfracture Luminosity', 
                         min_value=float(df['Sum_of_Luminosity'].min()),
                         max_value=float(df['Sum_of_Luminosity'].max()),
                         value=float(df['Sum_of_Luminosity'].median()))
    
    # Location data
    x_location = st.slider('X Coordinate', 
                         min_value=float(df['X_Minimum'].min()),
                         max_value=float(df['X_Minimum'].max()),
                         value=float(df['X_Minimum'].mean()))
    
    y_location = st.slider('Y Coordinate',
                         min_value=float(df['Y_Minimum'].min()),
                         max_value=float(df['Y_Minimum'].max()),
                         value=float(df['Y_Minimum'].mean()))
