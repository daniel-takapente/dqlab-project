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

# Menggunakan slider untuk input
sepal_length = st.slider('Panjang Sepal', min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.slider('Lebar Sepal', min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.slider('Panjang Petal', min_value=0.0, max_value=10.0, value=1.5)
petal_width = st.slider('Lebar Petal', min_value=0.0, max_value=10.0, value=0.3)
sepal_length = st.slider('Panjang Sepal', min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.slider('Lebar Sepal', min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.slider('Panjang Petal', min_value=0.0, max_value=10.0, value=1.5)
petal_width = st.slider('Lebar Petal', min_value=0.0, max_value=10.0, value=0.3)
sepal_length = st.slider('Panjang Sepal', min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.slider('Lebar Sepal', min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.slider('Panjang Petal', min_value=0.0, max_value=10.0, value=1.5)
petal_width = st.slider('Lebar Petal', min_value=0.0, max_value=10.0, value=0.3)

# Tombol untuk menjalankan prediksi
if st.button('Prediksi'):
    # Membuat array input
    input_data = [sepal_length, sepal_width, petal_length, petal_width]
    # Melakukan prediksi
    result = predict(input_data)
    st.write(f"Hasil Prediksi: {result}")
