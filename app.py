import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import KBinsDiscretizer

# Read the data
url = 'https://raw.githubusercontent.com/muhammadFandi12/mini-Project-Data-Mining/main/Data_Cleaning.csv'
df = pd.read_csv(url)

# Preprocess the data to ensure binary values
discretizer = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='uniform')
df_discretized = pd.DataFrame(discretizer.fit_transform(df), columns=df.columns)

st.set_option('deprecation.showPyplotGlobalUse', False)

with open('apple_model.pkl', 'rb') as f:
    apple_model = pickle.load(f)

# Set page title and icon
st.set_page_config(page_title="Data Mining", page_icon="📊")

# Define function to create navbar
def navbar():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ( "Home","EDA", "Association Rule Modeling"))
    return page

# Main function to display content based on navbar selection
def main():
    page = navbar()
    if page == "Home":
        st.title("Apple Quality")
        st.image("Apple.jpg")
        st.header("Business Understanding")
        st.write("""
        Tujuan dari Bisnis ini adalah Meningkatkan kualitas apel untuk meningkatkan kepuasan pelanggan dan daya saing pasar. Ini mencakup peningkatan kesegaran, rasa, dan nilai gizi apel, serta pengurangan cacat dan kerusakan.
        """)


        # Assess Situation
        st.header("Assess Situation")
        st.write("""
        Kualitas apel saat ini mungkin tidak konsisten di pasar. Pelanggan semakin memprioritaskan produk segar dan berkualitas tinggi, dan pesaing mungkin sudah menginvestasikan kontrol kualitas. Rantai pasokan apel harus dianalisis untuk mengidentifikasi titik potensial degradasi kualitas, mulai dari panen hingga distribusi.
        """)


        # Data Mining Goals
        st.header("Data Mining Goals")
        st.write("""
        Data tentang kualitas apel harus dikumpulkan dari pemasok, distributor, dan umpan balik pelanggan. Analisis data menggunakan metode statistik dan pembelajaran mesin untuk mengidentifikasi pola dan faktor yang memengaruhi kualitas apel. Model prediktif dapat dikembangkan untuk memprediksi kualitas apel berdasarkan faktor-faktor seperti kondisi panen, transportasi, dan metode penyimpanan. Wawasan yang diperoleh dari model harus dapat diubah menjadi strategi yang dapat diambil tindakan.
        """)

        # Project Plan
        st.header("Project Plan")
        st.write("""
        Proyek harus terdiri dari beberapa tahap: pengumpulan data, analisis data, pembuatan model, dan implementasi strategi. Setiap tahap harus diberi waktu yang cukup untuk menyelesaikannya dengan baik. Setelah model selesai, strategi berdasarkan wawasan model harus diimplementasikan dan dievaluasi untuk efektivitasnya.
        """)
    elif page == "EDA":
        st.title("Exploratory Data Analysis (EDA)")
        st.write("Performing EDA on the dataset...")
        st.write("### Dataset Overview")
        st.write(df.head())

        st.write("### Summary Statistics")
        st.write(df.describe())
        st.write("tabel menunjukkan bahwa berat rata-rata apel dalam kumpulan data adalah -1.0024 gram, dengan deviasi standar 1.5096 gram. Berat minimum adalah -5.0587 gram, dan berat maksimum adalah 3.0815 gram.")

        st.write("### Distribution of Numerical Features")
        plt.figure(figsize=(12, 6))
        df.select_dtypes(include=['float64', 'int64']).hist(bins=20, color='skyblue', edgecolor='black', linewidth=1.5)
        plt.tight_layout()
        st.pyplot()
        st.write("Gambar diatas menunjukkan distribusi dari enam fitur numerik: ukuran, berat, kemanisan, kerenyahan, kesegaran, dan keasaman. Plot distribusi memungkinkan Anda untuk menilai secara visual bagaimana data tersebar untuk setiap fitur. Misalnya, distribusi kemanisan tampak miring ke kanan, menunjukkan mungkin ada lebih banyak apel yang lebih manis daripada rata-rata.")

        st.write("### Correlation Matrix")
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.select_dtypes(include=['float64', 'int64']).corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        st.pyplot()


    elif page == "Association Rule Modeling":
            st.title('Data Mining APPLE QUALITY')

            A_id = st.text_input('Input nilai A_id')
            Weight = st.text_input('Input nilai Weight')
            Sweetness = st.text_input('Input nilai Sweetness')
            Crunchiness = st.text_input('Input nilai Crunchiness')
            Juiciness = st.text_input('Input nilai Juiciness')
            Ripeness = st.text_input('Input nilai Ripeness')
            Acidity = st.text_input('Input nilai Acidity')
            Quality = st.text_input('Input nilai Quality')

            if st.button('Prediksi Kualitas Apel'):
                # Check if all input fields are filled
                if A_id and Weight and Sweetness and Crunchiness and Juiciness and Ripeness and Acidity and Quality:
                    # Prepare input for prediction
                    input_data = [[A_id, Weight, Sweetness, Crunchiness, Juiciness, Ripeness, Acidity, Quality,0,0]]
                    
                    # Perform prediction
                    apple_quality = apple_model.predict(input_data)
                    
                    # Map prediction to "good" or "bad"
                    prediction_result = "good" if apple_quality[0] == 1 else "bad"
                    
                    # Display prediction
                    st.write(f'Predicted apple quality: {prediction_result}')
                else:
                    st.error("Mohon isi semua nilai input sebelum melakukan prediksi.")

if __name__ == "__main__":
    main()
