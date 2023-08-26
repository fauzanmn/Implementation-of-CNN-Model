import streamlit as st
import logging
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the Keras model
model = load_model('s2augm.h5')

# Membolehkan file dengan ekstensi png, jpg, dan jpeg
ALLOWED_EXT = set(['jpg', 'jpeg', 'png'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

# Fungsi untuk memuat dan mempersiapkan gambar dengan ukuran yang sesuai
def read_image(filename):
    img = load_img(filename, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x

def predict(file):
    img = read_image(file)
    class_prediction = model.predict(img)
    classes_x = np.argmax(class_prediction, axis=1)
    if classes_x == 0:
        cuaca = "Berawan ☁️"
    elif classes_x == 1:
        cuaca = "Cerah ☀️"
    elif classes_x == 2:
        cuaca = "Cerah Berawan ⛅"
    elif classes_x == 3:
        cuaca = "Hujan 🌧️"
    else:
        cuaca = "Lainnya ❓"
    return cuaca, class_prediction[0]

def predict_and_show_results(file):
    cuaca, class_prediction = predict(file)
    # Styling hasil prediksi
    st.markdown(f"<p style='font-size: 28px; font-weight: 800;'>Hasil Prediksi: {cuaca}</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 28px; font-weight: 800;'>Nilai Prediksi: </p>", unsafe_allow_html=True)
    for idx, prob in enumerate(class_prediction):
        class_name = ["Berawan", "Cerah", "Cerah Berawan", "Hujan", "Lainnya"][idx]
        icon = "☀️" if class_name == "Cerah" else "⛅" if class_name == "Cerah Berawan" else "🌧️" if class_name == "Hujan" else "☁️" if class_name == "Berawan" else "❓"

        # Menggunakan HTML untuk atur font size dan font weight pada tulisan probabilitas
        st.markdown(f"<p style='font-size: 24px; font-weight: 800;'>{icon} {class_name}: {prob * 100:.0f} %</p>", unsafe_allow_html=True)

def main():
    logger.info("Web sedang diakses")
    st.set_page_config(
        page_title="Weather Prediction | CNN LeNet-5",
        page_icon="🌦️",
        layout="wide"
    )
    st.markdown("<h1 style='text-align: center; font-size: 48px; padding: 0;'>Implementasi Model CNN</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; font-size: 48px; padding-top: 1px; margin-bottom: 15px;'>LeNet-5</h1>", unsafe_allow_html=True)

    # Menggunakan div untuk mengatur padding antara tulisan dan box uploader
    st.markdown(
        "<div style='font-weight: 600; text-align: center; margin-bottom: 0px;'>Please Upload Images for CNN Model Testing</div>",
        unsafe_allow_html=True
    )
    
    file = st.file_uploader('', type=['jpg', 'jpeg', 'png'])
    
    # Jika ada gambar yang diunggah, tampilkan di halaman utama
    if file is not None:
        if allowed_file(file.name):
            # Menambahkan gaya CSS untuk mengatur tinggi gambar
            st.markdown(f"<style>img {{ max-width:450px; max-height: 330px; border-radius: 10px; display: block; margin: 0 auto; box-shadow: 0px 0px 50px rgba(0, 0, 0, 0.1); }}</style>", unsafe_allow_html=True)
            st.image(file, caption='Gambar yang diunggah', use_column_width=True, output_format='PNG')
            logger.info("User telah upload gambar")
    # Button Predict di bagian utama halaman
    predict_button = st.button("Predict")
    
    # Jika tombol "Predict" ditekan, tampilkan halaman prediksi terpisah
    if predict_button:
        predict_and_show_results(file=file)
        logger.info("User telah mendapatkan hasil prediksi")

    # Footer dengan informasi hak cipta dan padding
    st.markdown("<hr style='margin-top: 20px; margin-bottom: 10px;'>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; font-size: 18px; font-weight: 600;'>© Klasifikasi Gambar Cuaca Model CNN LeNet-5 | Miranda Sahfira Tuna 2023</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
