import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# Load the trained model
model = tf.keras.models.load_model('salary_regression_model.h5')

# Load the scaler and encoders
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

# Streamlit app
st.title("Estimated Salary Prediction App")    

# User input
geography = st.selectbox("Select Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Select Gender", label_encoder_gender.classes_)
age = st.slider("Select Age", 18, 92)
balance = st.number_input("Enter Balance", min_value=0)
credit_score = st.number_input("Select Credit Score")
tenure = st.slider("Select Tenure", 0, 10)
num_of_products = st.slider("Select Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
exited = st.selectbox("Exited", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited],
    'Geography': [geography]
})

# One-hot encode the Geography column
geo_encoded = onehot_encoder_geo.transform(input_data[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

# Drop original Geography column ONCE
input_data = input_data.drop(columns=['Geography'])

# Concatenate encoded columns
input_data = pd.concat([input_data, geo_encoded_df], axis=1)

input_data = input_data[scaler.feature_names_in_]
# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict the salary
predicted_salary = model.predict(input_data_scaled)
predicted_salary = predicted_salary[0][0]

st.write(f"Estimated Salary: ${predicted_salary:.2f}")