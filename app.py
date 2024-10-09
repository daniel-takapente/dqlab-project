import streamlit as st
import joblib
import numpy as np

# Load model dari file .pkl
model = joblib.load('bank_model.pkl')

# Fungsi prediksi
def predict(input_data):
    prediction = model.predict([input_data])
    return prediction

# Judul aplikasi
st.title("Prediksi Potensial Nasabah Melalui Direct Marketing")
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

# Tombol untuk menjalankan prediksi
if st.button('Prediksi'):
    # Membuat array input
    input_data = [age, job, marital, education, default, housing, loan, contact, month, 
              day_of_week, duration, poutcome]
    # Melakukan prediksi
    result = predict(input_data)
    st.write(f"Hasil Prediksi: {result}")
