import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import math

df = pd.read_csv("sample_dataset.csv")
transformasi_data = pd.read_csv("transformasi_data.csv")

st.title("Prediksi Rating Toko Online dalam Platform Shopee Mall")
st.markdown("Analisis ini dilakukan berdasarkan 661 toko baju online yang ada di shopee mall, penelitian menggunakan 8 variabel prediktor yakni jumlah produk, performa chat, jumlah pengikut, jumlah penilaian, lama toko bergabung di online shop, daerah asal, dan voucher toko. 8 variabel prediktor ini diteliti untuk diidentifikasi apakah mempunyai pengaruh terhadap variabel respon yakni rating toko yang dikalikan dengan jumlah penilaian.")
st.caption("Sumber data: https://shopee.co.id/mall/brands/11042849")

tab_1, tab_2 = st.tabs(["Eksplorasi Data", "Analisis Regresi Linear"])

with tab_1:
    # Analisis Eksplorasi Data
    st.write(df)

    # Distribusi Data
    st.subheader("Distribusi Data")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Jumlah Produk",
        "Performa Chat",
        "Pengikut",
        "Lama Bergabung",
        "Rating"
    ])

    with tab1:
        fig = px.box(df, y="jumlah_produk")
        st.write(fig)
        st.markdown("Distribusi data tidak simetris namun miring ke kanan (positive skewness). Data memiliki outliers dan nilai ekstrim.")

    with tab2:
        fig = px.box(df, y="performa_chat")
        st.write(fig)
        st.markdown("Distribusi data tidak simetris namun miring ke kiri (negative skewness). Data memiliki outliers.")

    with tab3:
        fig = px.box(df, y="pengikut")
        st.write(fig)
        st.markdown("Distribusi data tidak simetris namun miring ke kanan (positive skewness). Data memiliki outliers dan nilai ekstrim.")

    with tab4:
        fig = px.box(df, y="lama_bergabung")
        st.write(fig)
        st.markdown("Distribusi data tidak simetris namun miring ke kanan (positive skewness). Data tidak memiliki outliers dan nilai ekstrim.")

    with tab5:
        fig = px.box(df, y="rating*jumlah_penilaian")
        st.write(fig)
        st.markdown("Distribusi data tidak simetris namun miring ke kanan (positive skewness). Data memiliki outliers dan nilai ekstrim.")

    # Frekuensi Data Kategorik
    st.subheader("Frekuensi Data Kategorik")

    fig = px.pie(df, names="voucher_toko")
    st.write(fig)
    st.caption("Proporsi variabel Voucher Toko")
    st.markdown("Berdasarkan pie chart, mayoritas toko tidak memiliki voucher toko (54,4%) sejumlah 136 toko dari total 250 toko.")

    fig = px.bar(df, x="Provinsi")
    st.write(fig)
    st.caption("Proporsi variabel Provinsi")
    st.markdown("Dapat dilihat dari bar chart untuk daerah asal berdasarkan provinsi, mayoritas toko berlokasi di Provinsi Jawa Barat (42,4%) sejumlah 106 toko dari total 250 toko.")

    # Korelasi Antar Variabel
    st.subheader("Korelasi Antar Variabel")
    st.markdown("Korelasi dibawah ini dihitung dengan metode korelasi pearson :")
    fig = px.imshow(df.corr(), text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
    st.write(fig)
    st.markdown("""
    * Jumlah produk yang dijual memiliki hubungan yang tidak cukup kuat dengan rating toko  jumlah penilaian yaitu sebesar 26%
    * Performa seller dalam membalas pesan dari pelanggan memiliki hubungan yang cukup rendah dengan rating toko  jumlah penilaian yaitu sebesar 14%
    * Jumlah pengikut toko memiliki hubungan yang kuat dengan rating toko  jumlah penilaian 89%
    * Lama bergabung sebuah toko dengan shopee memiliki hubungan yang cukup rendah dengan rating toko  jumlah penilaian 16%
    """)

with tab_2:
    # Transformasi Data
    st.subheader("Transformasi Data")
    st.markdown("Transformasi dilakukan pada setiap variabel yang bukan variabel dummy. Berikut ini merupakan percobaan regresi dengan menggunakan 10 jenis transformasi, didapatkan hasil sebagai berikut :")
    st.table(transformasi_data)
    st.markdown("Dari tabel diatas, transformasi logaritma pada setiap variabel kecuali lama_bergabung menghasilkan nilai r-square yang paling baik dengan jumlah variabel signifikan terbanyak di antara transformasi yang lain. Sehingga kita menggunakan data tersebut untuk membuat model regresi pada tahap-tahap selanjutnya.")

    # Feature Selection
    st.subheader("Feature Selection")
    st.markdown("Dibawah ini adalah line chart yang menggambarkan hasil r-square yang didapatkan dari feature selection")
    col1, col2 = st.columns(2)

    with col1:
        image = Image.open('featureselection1.jpg')
        st.image(image)
    with col2:
        image = Image.open('featureselection2.jpg')
        st.image(image)
        
    st.markdown("Kesimpulannya, nilai alpha yang dipilih untuk feature selection adalah 0.25 sebab dari situ kita mendapatkan nilai adjusted r-square terbaik. Metode yang digunakan bisa dipilih secara bebas karena ketiganya menghasilkan hasil yang sama. Kali ini, metode yang dipilih adalah stepwise. Diperoleh 6 variabel yang paling berpengaruh yaitu: jumlah_produk, performa_chat, pengikut, lama_bergabung, Provinsi_Bangka Belitung, Provinsi_Jawa Tengah")

    # Model Regresi
    st.subheader("Model Regresi")
    st.image("model.png")

    # Kebaikan Model
    st.subheader("Kebaikan Model")
    kebaikan_model = pd.read_csv("model_summary.csv")
    st.table(kebaikan_model)
    st.markdown("Dari tabel di atas diperoleh informasi bahwa kemampuan variabel independen dalam menjelaskan variasi dalam variabel rating  jumlah penilaian adalah sebesar 87.09%. Berarti terdapat 12.91% (100%-87.09%) varians dalam variabel rating  jumlah penilaian dijelaskan oleh faktor lain. Secara keseluruhan, model yang terbentuk sudah baik.")

    # Uji Asumsi
    st.subheader("Uji Asumsi")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Autokorelasi",
        "Heteroskedastisitas",
        "Normalitas Error",
        "Multikolinearitas"
    ])

    with tab1:
        image = Image.open('uji asumsi/autokorelasi.jpg')
        st.image(image)
        st.markdown("Secara visual, data tersebar di sekitar nol dan tidak memiliki pola, dapat disimpulkan bahwa error independen terhadap urutan data.")
        st.markdown("Hasil Statistik Uji:  Berdasarkan tabel hasil uji Durbin Watson, diperoleh nilai d (2.120) > dU (1.834), sehingga H0 diterima artinya error independen (tidak ada autokorelasi positif). Selain itu, diperoleh juga bahwa nilai 4-d (1.879) > dU (1.834) sehingga H0 diterima yang artinya tidak ada autokorelasi negatif. Maka dapat disimpulkan bahwa tidak ada sama sekali autokorelasi pada model regresi.")
    with tab2:
        image = Image.open('uji asumsi/hetero.jpg')
        st.image(image)
        st.markdown("Secara visual, error tersebar di sekitar nol yang artinya varians error homogen.")
        st.markdown("")
    with tab3:
        image = Image.open('uji asumsi/normalitas.jpg')
        st.image(image)
        st.markdown("Hasil Statistik Uji: Didapatkan hasil uji statistik atau nilai Kolmogorov Smirnov sebesar 0.095 dan nilai Asymp. Sig. (2-tailed) sebesar 0.000. Oleh karena nilai Asymp. Sig. (2-tailed) lebih kecil daripada alpha, maka kita dapat menyimpulkan error berdistribusi tidak normal. Hal tersebut tidak memenuhi asumsi regresi linear. Untuk mengatasinya, nilai outlier dapat dihapus terlebih dahulu dan menambahkan jumlah observasi.")
    with tab4:
        st.markdown("Deteksi multikolinearitas dapat dilakukan dengan mencari nilai Variance Inflation Factor (VIF).")
        vif = pd.read_csv('uji asumsi/vif.csv')
        st.table(vif)
        st.markdown("tidak terdapat masalah multikolinearitas pada setiap variabel karena nilai VIF tidak ada yang diatas 10 ataupun kurang dari 0,1. Oleh sebab itu, model yang telah dibuat tidak melanggar asumsi multikolinearitas.")

# Sidebar
with st.sidebar:
    with st.form("form"):
        st.subheader("Kalkulator Prediksi Rating Toko Online")
        jumlah_produk = st.number_input("Jumlah Produk Dijual", min_value=0)
        performa_chat = st.number_input("Performa Chat", min_value=0.0, max_value=1.0)
        pengikut = st.number_input("Jumlah Pengikut Toko", min_value=0)
        lama_bergabung = st.number_input("Lama Bergabung (Tahun)", min_value=0)
        opt = [
            "Nanggroe Aceh Darussalam",
            "Sumatra Utara",
            "Sumatra Selatan",
            "Sumatra Barat",
            "Bengkulu",
            "Riau",
            "Kepulauan Riau",
            "Jambi",
            "Lampung",
            "Bangka Belitung",
            "Kalimantan Timur",
            "Kalimantan Barat",
            "Kalimantan Tengah",
            "Kalimantan Selatan",
            "Kalimantan Utara",
            "DKI Jakarta",
            "Banten",
            "Jawa Barat",
            "Jawa Tengah",
            "DI Yogyakarta",
            "Jawa Timur",
            "Bali",
            "Nusa Tenggara Barat",
            "Nusa Tenggara Timur",
            "Sulawesi Utara",
            "Sulawesi Barat",
            "Sulawesi Tengah",
            "Gorontalo",
            "Sulawesi Tenggara",
            "Sulawesi Selatan",
            "Maluku Utara",
            "Maluku",
            "Papua Barat",
            "Papua",
            "Papua Selatan",
            "Papua Tengah",
            "Papua Pegunungan"
        ]
        Provinsi = st.selectbox("Domisili Toko", options=opt)

        if Provinsi == "Bangka Belitung":
            bangka_belitung = 1
            jawa_tengah = 0
        elif Provinsi == "Jawa Tengah":
            bangka_belitung = 0
            jawa_tengah = 1
        else:
            bangka_belitung = 0
            jawa_tengah = 0

        jumlah_penilaian = st.number_input("Jumlah Penilaian", min_value=1)

        submit = st.form_submit_button("Hitung Rating!")
        if submit:
            jumlah_produk = math.log10(jumlah_produk + 1)
            # st.write(jumlah_produk)
            performa_chat = math.log10(performa_chat + 1)
            # st.write(performa_chat)
            pengikut = math.log10(pengikut + 1)
            # st.write(pengikut)

            rating = -0.716 + 0.1166 * jumlah_produk + 1.314 * performa_chat + 1.0338 * pengikut + 0.0402 * lama_bergabung + 0.479 * bangka_belitung - 0.253 * jawa_tengah
            # st.write(rating)
            rating = 10**rating
            rating = rating / jumlah_penilaian
            rating = round(rating, 2)

            if rating > 5:
                rating = 5

            st.markdown(f"Berdasarkan prediksi, rating toko anda ketika produk terjual sejumlah {jumlah_penilaian} adalah {rating}/5")
            
