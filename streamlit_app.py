import streamlit as st
import pandas as pd

st.title('🌍 Earthquake Risk Detector')
st.write("This app predicts steel plate fault types to help assess potential earthquake vulnerabilities.")

with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('https://raw.githubusercontent.com/malekbenhafed/final-project/master/Steel_Plates_Faults.csv')
  st.dataframe(df)
  st.write('**X**')
  df = pd.read_csv('Steel_Plates_Faults.csv')

  st.write('**X (Features)**')
  X_raw = df.drop(['Pastry', 'Z_Scratch', 'K_Scratch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults'], axis=1)
  st.dataframe(X_raw)

  st.write('**y (Target - Fault Types)**')
  y_raw = df[['Pastry', 'Z_Scratch', 'K_Scratch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']]
  st.dataframe(y_raw)
