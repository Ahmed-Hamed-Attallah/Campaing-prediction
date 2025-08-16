# pages/model_deployment.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from PIL import Image

# -------- Page config (must be first Streamlit call) --------
st.set_page_config(
    page_title="Campaign Prediction",
    page_icon="🔮",
    layout="wide",
)

st.header("Under Developing)
