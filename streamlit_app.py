import streamlit as st
import pandas as pd
st.title('Earthquake detector')

st.write('The goal is to detect possible earthquake risks before something goes wrong')
df = pd.read_csv('https://raw.githubusercontent.com/malekbenhafed/final-project/master/Steel_Plates_Faults.csv')
df
