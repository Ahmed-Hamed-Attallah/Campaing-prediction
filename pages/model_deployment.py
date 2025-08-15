import streamlit as st
import pandas as pd
import joblib
import time 
from sklearn.base import BaseEstimator , TransformerMixin
import pandas as pd
from PIL import Image

# Set page configration 
st.set_page_config(
    layout="wide",
    page_title = 'Campaign Prediction',
    page_icon = '🔮'
)

col1 , col2 = st.columns([1,1])
with col1 :

    image = Image.open('machine-learning-logo.png')
    st.image(image, use_column_width=True)



with col2 :
    # Display a welcome message
    st.write("# Welcome to My Streamlit App")
    st.write('''### Hello, this is a machine learning classification model trained on real data.It is used to predict whether a client will subscribe to a specific term deposit campaign or not''')

# Add a delay (in seconds)
time.sleep(5)

# Hide the welcome message
st.empty()
# class labelencoder
class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        # Create a mapping dictionary for encoding
        map_dict = {}
        for column in self.columns:
            map_dict[column] = {}
            unique_values = [value for value in X[column].unique() if value != 'unknown']
            for i, value in enumerate(unique_values):
                map_dict[column][value] = i
        
        self.map_dict = map_dict
        return self
    
    def transform(self, X):
        X_encoded = X.copy()
        for key in self.map_dict.keys():
            X_encoded[key] = X_encoded[key].map(self.map_dict[key])
        return X_encoded

# import
model = joblib.load('pages/LR_Model.pkl')
columns = joblib.load('pages/columns.pkl')


# define function for age 
def classify_age(age):
    if age < 18:
        return "Teenagers"
    elif age < 30:
        return "Youth"
    elif age < 45:
        return "Young Adults"
    elif age < 65:
        return "Adults"
    else:
        return "Elderly"

# define function for month 
def classify_season(month):
    if month in ['dec', 'jan', 'feb']:
        return "Winter"
    elif month in ['mar', 'apr', 'may']:
        return "Spring"
    elif month in ['jun', 'jul','aug']:
        return "Summer"
    elif month in ['sep','oct', 'nov']:
        return "Autumn"
    else:
        return "Invalid month"


def prediction(job, marital, education, housing, loan, contact,day_of_week,duration,campaign,pdays,previous,poutcom,emp,price,conf, euribor3m,employed,month, age):
    df = pd.DataFrame(columns = columns)
    df.at[ 0 , 'age_catigories' ] = classify_age(age)
    df.at[0, 'duration'] = duration
    df.at[0 , 'campaign'] = campaign
    df.at[0,'pdays'] = pdays
    df.at[0,'previous'] = previous
    df.at[0,'job'] = job
    df.at[0,'loan'] = loan
    df.at[0,'housing'] = housing
    df.at[0,'season'] = classify_season(month)
    df.at[0,'contact'] = contact
    df.at[0,'euribor3m'] = euribor3m
    df.at[0,'poutcom'] = poutcom
    df.at[0,'nr.employed'] = employed
    df.at[0,'emp.var.rate'] = emp
    df.at[0,'cons.price.idx'] = price
    df.at[0,'marital'] = marital
    df.at[0,'day_of_week'] = day_of_week
    df.at[0,'education'] = education
    df.at[0,'cons.conf.idx'] = conf
    
    res = model.predict(df)
    return res[0]


st.title("Pridect if the client will subscribe a term deposit")
col1 , col2 , col3 = st.columns([1,0.25 ,2])
with col1:
    age = st.slider('age' , min_value=12 , max_value=100 , step=1 , value=16 )
    duration =  st.slider("duration of call" ,min_value=0,max_value=90,step = 1 , value = 1 )
    campaign = st.slider("number of call during campign" ,min_value=1,max_value=60,step = 1 , value = 1 )
    emp = st.slider("employment variation rate" ,min_value= -4.0,max_value=2.0,step = 0.1 , value =-1.1 )
    pdays = st.slider('Number of days since the last contact' , min_value=0 , max_value=30 ,step=1 ,value=0)
    previous = st.slider('Number of previous contacts', min_value=0 , step = 1 , max_value=10 , value = 1)
    euribor3m = st.slider('Monthly Triple Interest Rate', min_value=0.0 , max_value=10.0 , step = 0.1)
    conf = st.slider('confidence interval' , min_value = -60.0 , step= 0.1 , max_value=-20.0 , value=-30.0)
    price = st.slider('rate of change in commodity prices' , min_value = 90.0 , step= 0.1 , max_value=100.0 , value=90.0)
    employed = st.slider('number of employee' , min_value=4900 , step=1,max_value=5250)

with col3 :
    job = st.selectbox('Type of job' , ['housemaid', 'services', 'admin.', 'blue-collar', 'technician',
       'retired', 'management', 'unemployed', 'self-employed', 'other',
       'entrepreneur', 'student'])
    marital = st.selectbox('Marital status' , ['married', 'single', 'divorced', 'unknown'])
    education= st.selectbox('Education degree',['basic.4y', 'high.school', 'basic.6y', 'basic.9y',
       'professional.course', 'other', 'university.degree', 'illiterate'])
    housing= st.selectbox('housing loan' , ['no', 'yes', 'unknown'])
    loan = st.selectbox('Personal loan',['no', 'yes', 'unknown']) 
    contact = st.selectbox('type contact' , ['telephone', 'cellular'])
    day_of_week = st.selectbox('day' , ['mon', 'tue', 'wed', 'thu', 'fri'])
    poutcom = st.selectbox('result of the previous campaign' , ['nonexistent', 'failure', 'success'])
    month = st.selectbox('Month', ['dec', 'jan', 'feb','mar', 'apr', 'may','jun', 'jul','aug','sep','oct', 'nov'])
    

if st.button("Predict"):
    result = prediction(job, marital, education, housing, loan, contact,day_of_week, duration, campaign,pdays,previous,poutcom,emp,price,conf, euribor3m,employed,month,age)
    res_list = ["No,he will not subscribe","yes,he will subscribe"]
    st.write(res_list[result])

