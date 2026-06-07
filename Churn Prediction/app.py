import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Get folder where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model
with open(os.path.join(BASE_DIR, "gb_model.pkl"), "rb") as file:
    model = pickle.load(file)

# Load scaler
with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as file:
    scaler = pickle.load(file)

# Load feature names
with open(os.path.join(BASE_DIR, "feature_names.pkl"), "rb") as file:
    feature_names = pickle.load(file)

st.title("Bank Customer Churn Prediction System")

st.write("Predict customer churn probability and risk level")

st.header("Enter Customer Details")
    
   
year= 2026
credit_score = st.number_input("Credit Score", 300, 900, 650)

age = st.number_input("Age", 18, 100, 35)

tenure = st.number_input("Tenure", 0, 10, 5)

balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)

num_products = st.number_input("Number of Products", 1, 4, 1)

has_card = st.selectbox("Has Credit Card", [0, 1])

is_active = st.selectbox("Is Active Member", [0, 1])

estimated_salary = st.number_input(
    "Estimated Salary",
    0.0,
    200000.0,
    50000.0
)

geography = st.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

# Prediction Button
if st.button("Predict Churn"):

    # Geography Encoding
    geo_germany = 1 if geography == "Germany" else 0
    geo_spain = 1 if geography == "Spain" else 0

    # Gender Encoding
    gender_male = 1 if gender == "Male" else 0

    # Feature Engineering
    balance_salary_ratio = balance / (estimated_salary + 1)

    product_density = num_products / (tenure + 1)

    engagement_product = is_active * num_products

    age_tenure = age * tenure

    year=2026
    # Create Input DataFrame
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_products],
        'HasCrCard': [has_card],
        'IsActiveMember': [is_active],
        'EstimatedSalary': [estimated_salary],
        'Geography_Germany': [geo_germany],
        'Geography_Spain': [geo_spain],
        'Gender_Male': [gender_male],
        'BalanceSalaryRatio': [balance_salary_ratio],
        'ProductDensity': [product_density],
        'EngagementProduct': [engagement_product],
        'AgeTenure': [age_tenure],
        'Year': [year]
    })

    input_data = input_data[feature_names]
    
    # Predict probability
    probability = model.predict_proba(input_data)[0][1]

    # Final Prediction
    prediction = model.predict(input_data)[0]

    # Risk Category
    if probability < 0.30:
        risk = "Low Risk"

    elif probability < 0.60:
        risk = "Medium Risk"

    else:
        risk = "High Risk"

    # Display Results
    st.subheader("Prediction Result")

    st.write(f"Churn Probability: {probability:.2f}")

    st.write(f"Risk Category: {risk}")

    if prediction == 1:
        st.error("Customer is likely to churn.")

    else:
        st.success("Customer is likely to stay.")
        