import streamlit as st
import pandas as pd

st.title('🌍 Earthquake Risk Detector')
st.write("This app predicts steel plate fault types to help assess potential earthquake vulnerabilities.")

with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('https://raw.githubusercontent.com/malekbenhafed/final-project/master/Steel_Plates_Faults.csv')
  st.dataframe(df)
  st.write('**X**')
  X = df.drop('species', axis=1)
  X

  st.write('**y**')
  y= df.species
  y
