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

def AUD_to_EGP(x):
    response = req.get("https://www.exchangerates.org.uk/Egyptian-Pound-EGP-currency-table.html")
    soup = BeautifulSoup(response.text , "html.parser")
    price = float(soup.find('div' , attrs={'class' : 'large-8 medium-6 small-12 columns nolpad'}).find_all('tr' ,
                                    attrs={'class' : 'colone'})[0].find('strong').text)
    today =f'1 AUD today = {price} EGP'
    reslut =f'{x} AUD = {round(x*price , 2)}'
    return today , reslut

def GBP_to_EGP(x):
    response = req.get("https://www.exchangerates.org.uk/Egyptian-Pound-EGP-currency-table.html")
    soup = BeautifulSoup(response.text , "html.parser")
    price = float(soup.find('div' , attrs={'class' : 'large-8 medium-6 small-12 columns nolpad'}).find_all('tr' ,
                                    attrs={'class' : 'coltwo'})[0].find('strong').text)
    today = f'1 GBP today = {price} EGP'
    result = f'{x} GBP = {round(x*price , 2)}'
    return today , result

def USD_to_EGP(x):
    response = req.get("https://www.exchangerates.org.uk/Egyptian-Pound-EGP-currency-table.html")
    soup = BeautifulSoup(response.text , "html.parser")
    price = float(soup.find('div' , attrs={'class' : 'large-8 medium-6 small-12 columns nolpad'}).find_all('tr' ,
                                    attrs={'class' : 'colone'})[2].find('strong').text)
    today = f'1 USD today = {price} EGP'
    result = f'{x} USD = {round(x*price , 2)}'
    return today , result

def EUR_to_EGP(x):
    response = req.get("https://www.exchangerates.org.uk/Egyptian-Pound-EGP-currency-table.html")
    soup = BeautifulSoup(response.text , "html.parser")
    price = float(soup.find('div' , attrs={'class' : 'large-8 medium-6 small-12 columns nolpad'}).find_all('tr' ,
                                    attrs={'class' : 'colone'})[1].find('strong').text)
    today = f'1 EUR today = {price} EGP'
    result = f'{x} EUR = {round(x*price , 2)}'
    return today , result

def NZD_to_EGP(x):
    response = req.get("https://www.exchangerates.org.uk/Egyptian-Pound-EGP-currency-table.html")
    soup = BeautifulSoup(response.text , "html.parser")
    price = float(soup.find('div' , attrs={'class' : 'large-8 medium-6 small-12 columns nolpad'}).find_all('tr' ,
                                    attrs={'class' : 'coltwo'})[1].find('strong').text)
    today = f'1 NZD today = {price} EGP'
    result  = f'{x} NZD = {round(x*price , 2)}'
    return today , result

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

    calc_type = st.selectbox('pleas select Currency from this box :' , ['Australian Dollars to Egyptian Pounds' ,
                                                           'Pounds to Egyptian Pounds' , 'Euros to Egyptian Pounds' ,
                                                           'New Zealand Dollars to Egyptian Pounds' , 'Dollars to Egyptian Pounds'])

    if calc_type == 'Australian Dollars to Egyptian Pounds' :
        aud = st.number_input('pleas enter number of Australian Dollars')
        st.text(AUD_to_EGP(aud))
    elif calc_type== 'Pounds to Egyptian Pounds' :
        gbp = st.number_input('pleas enter number of Pounds')
        st.text(GBP_to_EGP(gbp))
    elif calc_type == 'Euros to Egyptian Pounds' :
        eur = st.number_input('pleas enter number Euros')
        st.text(EUR_to_EGP(eur))
    elif calc_type == 'New Zealand Dollars to Egyptian Pounds' :
        nzd = st.number_input('pleas enter number of New Zealand Dollars' )
        st.text(NZD_to_EGP(nzd))
    else :
        usd =  st.number_input('pleas enter number of Dollars')
        st.text(USD_to_EGP(usd))
