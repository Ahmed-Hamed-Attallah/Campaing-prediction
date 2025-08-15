import streamlit as st
import pandas as pd
import requests as req
from bs4 import BeautifulSoup
import joblib as jo


df = pd.read_csv('Bank_data.csv')
df['duration'] = df['duration'].apply(lambda x : x/60)
def x(x):
    if x == 999 :
        return 0
    else :
        return x
df['pdays'] = df['pdays'].apply(x)

st.set_page_config(
        layout = 'wide',
        page_title = 'DashBoard',
        page_icon= '📊'
)



tab1 , tab2 = st.tabs(['sample of Dataset' , 'Currency calculator 🧮'])

with tab1 :
    st.markdown('<h3 style="text-align: center; color :#22A8A4;">Sample of Dataset</h3>', unsafe_allow_html=True)
    st.dataframe(df.sample(1500).reset_index(drop=True))
    st.text('count of this sample = 1500')
    col1 , col2 = st.columns([1,1])
    with col1 :
        st.markdown('<h3 style="text-align: center; color :red;">Nomerical Descriptive Stat </h3>', unsafe_allow_html=True)
        st.dataframe(df.describe() , use_container_width=True)
    with col2 :
        st.markdown('<h3 style="text-align: center; color :red;">Categorical Descriptive Stat</h3>', unsafe_allow_html=True)
        st.dataframe(df.describe(include='O') , use_container_width=True)

with tab2 :
    st.markdown('<h3 style="text-align: center; color :#1450DB;">Currency calculator 🧮</h3>', unsafe_allow_html=True)
    st.markdown('<h5 style="text-align: center; color :#1450DB;">Under Working</h5>', unsafe_allow_html=True)
   
    
