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

# -------- Custom objects that might be needed for unpickling --------
class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.map_dict = {}

    def fit(self, X, y=None):
        map_dict = {}
        for column in self.columns:
            map_dict[column] = {}
            unique_values = [value for value in X[column].unique() if value != "unknown"]
            for i, value in enumerate(unique_values):
                map_dict[column][value] = i
        self.map_dict = map_dict
        return self

    def transform(self, X):
        X_encoded = X.copy()
        for key in self.map_dict.keys():
            if key in X_encoded.columns:
                X_encoded[key] = X_encoded[key].map(self.map_dict[key])
        return X_encoded



# -------- Helpers --------
@st.cache_resource(show_spinner=False)
def load_model_and_columns():
    try:
        model = joblib.load("LR_Model.pkl")  # relies on LabelEncoderTransformer being defined if used in the pipeline
    except AttributeError as e:
        st.error(
            "Model file found but could not be loaded due to a missing class or version mismatch.\n\n"
            f"Details: {e}\n\n"
            "Make sure any custom transformers used during training (e.g., LabelEncoderTransformer) "
            "are defined in this file and that scikit-learn/joblib versions match the training environment."
        )
        st.stop()
    except FileNotFoundError:
        st.error("Model file 'LR_Model.pkl' not found in the app directory.")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error while loading the model: {e}")
        st.stop()

    try:
        columns = joblib.load("columns.pkl")
        if not isinstance(columns, (list, pd.Index)):
            raise ValueError("columns.pkl must contain a list-like of column names.")
        columns = list(columns)
    except FileNotFoundError:
        st.error("Columns file 'columns.pkl' not found in the app directory.")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error while loading 'columns.pkl': {e}")
        st.stop()

    return model, columns

def classify_age(age: int) -> str:
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

def classify_season(month: str) -> str:
    if month in ["dec", "jan", "feb"]:
        return "Winter"
    elif month in ["mar", "apr", "may"]:
        return "Spring"
    elif month in ["jun", "jul", "aug"]:
        return "Summer"
    elif month in ["sep", "oct", "nov"]:
        return "Autumn"
    return "Invalid month"

def build_feature_row(
    job, marital, education, housing, loan, contact, day_of_week,
    duration, campaign, pdays, previous, poutcom, emp, price, conf,
    euribor3m, employed, month, age, columns
) -> pd.DataFrame:
    df = pd.DataFrame(columns=columns)

    # Map raw inputs into your feature schema (keep your original keys/names)
    df.at[0, "age_catigories"] = classify_age(age)
    df.at[0, "duration"] = duration
    df.at[0, "campaign"] = campaign
    df.at[0, "pdays"] = pdays
    df.at[0, "previous"] = previous
    df.at[0, "job"] = job
    df.at[0, "loan"] = loan
    df.at[0, "housing"] = housing
    df.at[0, "season"] = classify_season(month)
    df.at[0, "contact"] = contact
    df.at[0, "euribor3m"] = euribor3m
    df.at[0, "poutcom"] = poutcom           # keep exact spelling used in training
    df.at[0, "nr.employed"] = employed
    df.at[0, "emp.var.rate"] = emp
    df.at[0, "cons.price.idx"] = price
    df.at[0, "marital"] = marital
    df.at[0, "day_of_week"] = day_of_week
    df.at[0, "education"] = education
    df.at[0, "cons.conf.idx"] = conf

    # Ensure numeric columns are numeric (if they exist in the schema)
    numeric_candidates = [
        "duration", "campaign", "pdays", "previous", "euribor3m",
        "nr.employed", "emp.var.rate", "cons.price.idx", "cons.conf.idx"
    ]
    for c in numeric_candidates:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

# -------- UI Header --------
header_col1, header_col2 = st.columns([1, 1])
with header_col1:
    try:
        image = Image.open("machine-learning-logo.png")
        st.image(image, use_column_width=True)
    except Exception:
        st.write("")

with header_col2:
    st.write("# Welcome to the Campaign Prediction App")
    st.write(
        "### This classification model predicts whether a client will subscribe to a term deposit campaign."
    )

st.markdown("---")

# -------- Load model & columns once --------
model, columns = load_model_and_columns()

# -------- Inputs --------
st.title("Predict if the client will subscribe to a term deposit")

left, spacer, right = st.columns([1, 0.2, 2])

with left:
    age = st.slider("Age", min_value=12, max_value=100, step=1, value=25)
    duration = st.slider("Duration of call (seconds)", min_value=0, max_value=5000, step=5, value=60)
    campaign = st.slider("Number of calls during the campaign", min_value=1, max_value=60, step=1, value=1)
    emp = st.slider("Employment variation rate", min_value=-4.0, max_value=2.0, step=0.1, value=-1.1)
    pdays = st.slider("Days since last contact", min_value=0, max_value=999, step=1, value=0)
    previous = st.slider("Number of previous contacts", min_value=0, max_value=10, step=1, value=0)
    euribor3m = st.slider("Euribor 3m", min_value=0.0, max_value=10.0, step=0.1, value=3.5)
    conf = st.slider("Consumer confidence index", min_value=-60.0, max_value=-20.0, step=0.1, value=-30.0)
    price = st.slider("Consumer price index", min_value=90.0, max_value=110.0, step=0.1, value=93.0)
    employed = st.slider("Number of employees", min_value=4900, max_value=5250, step=1, value=5000)

with right:
    job = st.selectbox(
        "Job",
        [
            "housemaid", "services", "admin.", "blue-collar", "technician",
            "retired", "management", "unemployed", "self-employed", "other",
            "entrepreneur", "student",
        ],
    )
    marital = st.selectbox("Marital status", ["married", "single", "divorced", "unknown"])
    education = st.selectbox(
        "Education",
        [
            "basic.4y", "high.school", "basic.6y", "basic.9y",
            "professional.course", "other", "university.degree", "illiterate",
        ],
    )
    housing = st.selectbox("Housing loan", ["no", "yes", "unknown"])
    loan = st.selectbox("Personal loan", ["no", "yes", "unknown"])
    contact = st.selectbox("Contact type", ["telephone", "cellular"])
    day_of_week = st.selectbox("Day of week", ["mon", "tue", "wed", "thu", "fri"])
    poutcom = st.selectbox("Previous campaign outcome", ["nonexistent", "failure", "success"])
    month = st.selectbox("Last contact month", ["dec", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov"])

# -------- Predict --------
if st.button("Predict"):
    # Build features row aligned to training schema
    row = build_feature_row(
        job, marital, education, housing, loan, contact, day_of_week,
        duration, campaign, pdays, previous, poutcom, emp, price, conf,
        euribor3m, employed, month, age, columns
    )

    try:
        pred = model.predict(row)
        # Many sklearn classifiers return numpy arrays; ensure we handle 0/1 or labels
        val = pred[0]
        # Map 0/1 -> message; if it's already "yes"/"no" strings, handle gracefully
        mapping = {0: "No, the client will not subscribe.", 1: "Yes, the client will subscribe."}
        output = mapping[val] if val in mapping else str(val)
        st.success(output)
        st.caption("Prediction generated based on the provided inputs.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

