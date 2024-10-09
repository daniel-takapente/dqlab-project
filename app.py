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
col1, col2 = st.columns(2)

with col1:
# Creating input fields for new parameters
	age = st.slider('Umuer', min_value=18, max_value=100, value=30)
	job = st.selectbox('Pekerjaan', ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 
	                             'retired', 'student', 'technician', 'unemployed', 'unknown'])
	marital = st.selectbox('Status Pernikahan', ['divorced', 'married', 'single', 'unknown'])
	education = st.selectbox('Tingkat Pendidikan', ['primary', 'secondary', 'tertiary', 'unknown'])
	default = st.selectbox('Memiliki Kredit ?', ['yes', 'no'])
	housing = st.selectbox('Memiliki KPR?', ['yes', 'no'])

with col2:
	loan = st.selectbox('Memiliki Personal Loan?', ['yes', 'no'])
	contact = st.selectbox('Alat Komunikasi ', ['cellular', 'telephone', 'unknown'])
	month = st.selectbox('Bulan Terakhid Di hubungi', ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
	                                                    'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
	day_of_week = st.selectbox('Hari terakhir di hubungi', ['mon', 'tue', 'wed', 'thu', 'fri'])
	duration = st.number_input('Durasi percakapan terakhir ', min_value=0, value=0)
	poutcome = st.selectbox('Hasil Kampanye marketing sebelumnya', 
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
