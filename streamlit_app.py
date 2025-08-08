import streamlit as st
import pandas as pd

st.title('🌍 Earthquake Risk Detector')
st.write("This app predicts steel plate fault types to help assess potential earthquake vulnerabilities.")

df = pd.read_csv('https://github.com/malekbenhafed/final-project')
df
