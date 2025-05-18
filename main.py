import pandas as pd
import numpy as np
import streamlit as st
import pickle

# Load trained model
with open("xgboost_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load expected feature names in the correct order
with open("selected_features.pkl", "rb") as f:
    expected_features = pickle.load(f)

# Define mappings for categorical variables
marital_mapping = {
    'married': 0,
    'single': 1,
    'divorced': 2,
    'unknown': 3
}

education_mapping = {
    'basic.4y': 0,
    'basic.6y': 1,
    'basic.9y': 2,
    'high.school': 3,
    'illiterate': 4,
    'professional.course': 5,
    'university.degree': 6,
    'unknown': 7
}

contact_mapping = {
    'cellular': 0,
    'telephone': 1
}

poutcome_mapping = {
    'nonexistent': 0,
    'failure': 1,
    'success': 2
}

# Streamlit UI
st.title("📈 Term Deposit Subscription Predictor")

st.write("Please enter customer information:")

# Input fields for the expected 15 features
age = st.slider("Age", 18, 100, 40)
marital = st.selectbox("Marital Status", list(marital_mapping.keys()))
education = st.selectbox("Education", list(education_mapping.keys()))
contact = st.selectbox("Contact Communication Type", list(contact_mapping.keys()))
duration = st.slider("Last Contact Duration (seconds)", 0, 5000, 100)
campaign = st.slider("Number of Contacts in Campaign", 1, 50, 1)
pdays = st.slider("Days Since Last Contact", -1, 1000, 999)
previous = st.slider("Number of Contacts Before Campaign", 0, 10, 0)
poutcome = st.selectbox("Outcome of Previous Campaign", list(poutcome_mapping.keys()))
emp_var_rate = st.slider("Employment Variation Rate", -3.0, 2.0, 0.0, step=0.1)
cons_price_idx = st.slider("Consumer Price Index", 92.0, 95.0, 93.0, step=0.001)  # ✅ ADDED
cons_conf_idx = st.slider("Consumer Confidence Index", -50.0, 0.0, -40.0, step=0.1)
euribor3m = st.slider("Euribor 3 Month Rate", 0.0, 6.0, 1.0, step=0.01)
nr_employed = st.slider("Number of Employees", 4960.0, 5220.0, 5000.0, step=1.0)
economic_index = st.slider("Economic Sentiment Index", 0.0, 1.0, 0.5, step=0.01)

# Construct input dictionary
input_dict = {
    "age": age,
    "marital": marital_mapping[marital],
    "education": education_mapping[education],
    "contact": contact_mapping[contact],
    "duration": duration,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": poutcome_mapping[poutcome],
    "emp.var.rate": emp_var_rate,
    "cons.price.idx": cons_price_idx,  # ✅ ADDED
    "cons.conf.idx": cons_conf_idx,
    "euribor3m": euribor3m,
    "nr.employed": nr_employed,
    "economic_index": economic_index
}

# Ensure feature order matches training
input_data = pd.DataFrame([{feature: input_dict[feature] for feature in expected_features}])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("✅ The model predicts that the client is **likely to subscribe** to a term deposit.")
    else:
        st.warning("❌ The model predicts that the client is **unlikely to subscribe** to a term deposit.")
