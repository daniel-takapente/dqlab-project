import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# Load the trained model and scaler
model = joblib.load('bank_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define the prediction function
def predict(input_data):
    # Scale the input data
    input_scaled = scaler.transform(np.array(input_data).reshape(1, -1))  # Make it 2D
    # Use the model to make a prediction
    prediction = model.predict(input_scaled)
    return prediction

# Create the Streamlit user interface
st.title("Bank Marketing Prediction App")

# Creating input fields for new parameters
age = st.slider('Age', min_value=18, max_value=100, value=30)
job = st.selectbox('Job', ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 
                             'retired', 'student', 'technician', 'unemployed', 'unknown'])
marital = st.selectbox('Marital Status', ['divorced', 'married', 'single', 'unknown'])
education = st.selectbox('Education Level', ['primary', 'secondary', 'tertiary', 'unknown'])
default = st.selectbox('Credit in Default?', ['yes', 'no'])
housing = st.selectbox('Housing Loan?', ['yes', 'no'])
loan = st.selectbox('Personal Loan?', ['yes', 'no'])
contact = st.selectbox('Contact Communication Type', ['cellular', 'telephone', 'unknown'])
month = st.selectbox('Last Contact Month of Year', ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                                                    'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
day_of_week = st.selectbox('Last Contact Day of Week', ['mon', 'tue', 'wed', 'thu', 'fri'])
duration = st.number_input('Duration of Last Contact (seconds)', min_value=0, value=0)
poutcome = st.selectbox('Outcome of the Previous Marketing Campaign', 
                         ['failure', 'nonexistent', 'success'])

# Collect input data into a list
input_data = [age, job, marital, education, default, housing, loan, contact, month, 
              day_of_week, duration, poutcome]

# Button to trigger prediction
if st.button('Predict'):
    # Make a prediction
    prediction = predict(input_data)
    
    # Map the prediction to "Yes" or "No"
    if prediction[0] == 1:
        st.write('Prediction: Yes, the user is likely to subscribe to a term deposit.')
    else:
        st.write('Prediction: No, the user is unlikely to subscribe to a term deposit.')
