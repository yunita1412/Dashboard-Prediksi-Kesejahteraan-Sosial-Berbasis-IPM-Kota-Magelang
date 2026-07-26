import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import os

st.set_page_config(
    page_title="Dashboard Prediksi Kesejahteraan Sosial Berbasis IPM Kota Magelang",
    layout="wide"
)

st.markdown("""
<style>
div[data-testid="stMetric"]{
    background-color:#EAF2FF;
    border-left:6px solid #F4B400;
    padding:15px;
    border-radius:10px;
}

h1,h2,h3{
    color:#0F4C81;
}

section[data-testid="stSidebar"]{
    background-color:#0F4C81;
}

section[data-testid="stSidebar"] *{
    color:white;
}
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.join(
    BASE_DIR, "dataset_final.xlsx"
)

FORECAST_IPM = os.path.join(
    BASE_DIR, "forecast_ipm.xlxs"
)

MEAN_SHAP_PATH = os.path.join(
    BASE_DIR, "mean_shap.xlsx"
)

FEATURE_IMPORTANCE_PATH = os.path.join(
    BASE_DIR, "feature_importance.xlsx"
)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset tidak ditemukan: {DATA_PATH}"
    )

@st.cache_data
def load_data():
    return pd.read_excel(DATA_PATH)

df = load_data()

menu = st.sidebar.radio(
    "Menu",
    ["Beranda","Prediksi IPM","Analisis Faktor","Informasi Sistem"]
)

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

forecast = pd.read_excel(
    os.path.join("dataset_xgboost.xlsx")
)

evaluasi_prophet = pd.read_excel(
    os.path.join("evaluasi_prophet.xlsx")
)

evaluasi_xgboost = pd.read_excel(
    os.path.join("evaluasi_xgboost.xlsx")
)

mean_shap = pd.read_excel(
    os.path.join("mean_shap.xlsx")
)

feature_importance = pd.read_excel(
    os.path.join("feature_importance.xlsx")
)

top5 = mean_shap.head(5).reset_index(drop=True)

if menu == "Beranda":

    st.title("Dashboard Prediksi Kesejahteraan Sosial Berbasis IPM Kota Magelang")

    st.info("""
    Dashboard ini digunakan untuk membantu memantau,
    memprediksi, dan menganalisis faktor yang mempengaruhi
    Indeks Pembangunan Manusia (IPM) Kota Magelang.
    """)


    st.subheader("Informasi Dataset")
    df["Tahun"] = pd.to_datetime(df["ds"]).dt.year

    c1, c2, c3  = st.columns(3)

    c1.metric("Jumlah Data", len(df))
    c2.metric("Jumlah Variabel", df.shape[1])
    c3.metric("Periode Data", f"{df['Tahun'].min()} - {df['Tahun'].max()}")

    st.markdown("---")

    st.subheader("Dataset")

    st.dataframe(
        df,
        use_container_width=True
    )

elif menu == "Prediksi IPM":

    st.title("Prediksi Indeks Pembangunan Manusia (IPM)")

    forecast = pd.read_excel(os.path.join
        ("forecast_ipm.xlsx")
    )
    forecast["ds"] = pd.to_datetime(forecast["ds"])

    tahun_prediksi = sorted(
    forecast.loc[
        forecast["ds"].dt.year >= 2026,
        "ds"
    ].dt.year.unique())

    tahun_pilihan = st.selectbox(
        "Pilih Tahun Prediksi",
        tahun_prediksi
    )

    forecast_tahun = forecast[
        forecast["ds"].dt.year == tahun_pilihan
    ]

    prediksi_ipm = forecast_tahun["IPM_Prediksi_Prophet"].mean()

    forecast_plot = forecast.copy()
    forecast_plot["Tahun"] = forecast_plot["ds"].dt.year

    pred_tahun = (
        forecast_plot.groupby("Tahun")["IPM_Prediksi_Prophet"]
        .mean()
        .reset_index()
    )

    aktual = (
        df.groupby("Tahun")["IPM"]
        .mean()
        .reset_index()
    )

    aktual["Jenis"] = "Aktual"
    aktual.columns = ["Tahun", "Nilai", "Jenis"]

    forecast_plot = forecast.copy()
    forecast_plot["Tahun"] = forecast_plot["ds"].dt.year

    pred_tahun = (
        forecast_plot.groupby("Tahun")["IPM_Prediksi_Prophet"]
        .mean()
        .reset_index()
    )

    prediksi = pred_tahun.copy()
    prediksi["Jenis"] = "Prediksi"
    prediksi.columns = ["Tahun", "Nilai", "Jenis"]

    prediksi = prediksi[
        prediksi["Tahun"] <= tahun_pilihan
    ]

    gabung = pd.concat([aktual, prediksi])

    fig = px.line(
        gabung,
        x="Tahun",
        y="Nilai",
        color="Jenis",
        markers=True,
        title=f"IPM Aktual dan Prediksi hingga Tahun {tahun_pilihan}"
    )

    fig.update_traces(
        selector=dict(name="Prediksi"),
        line=dict(color="#FF0000"),
        marker=dict(color="#FF0000", size=5)
    )

    fig.update_traces(
        selector=dict(name="Aktual"),
        line=dict(color="#FFD700"),
        marker=dict(color="#FFD700", size=10)
    )

    fig.update_layout(
        xaxis_title="Tahun",
        yaxis_title="IPM"
    )

    fig.update_xaxes(
        tickmode="linear",
        tick0=2010,
        dtick=2
    )

    st.plotly_chart(fig, use_container_width=True)

    pred_tahun.columns = ["Tahun", "Prediksi"]

    if tahun_pilihan <= df["Tahun"].max():
        pembanding = df[df["Tahun"] == tahun_pilihan - 1]["IPM"].values[0]
    else:
        pembanding = pred_tahun.loc[
            pred_tahun["Tahun"] == tahun_pilihan - 1,
            "Prediksi"
        ].values[0]

    selisih = prediksi_ipm - pembanding

    if prediksi_ipm >= 80:
        kategori = "Sangat Tinggi"
    elif prediksi_ipm >= 70:
        kategori = "Tinggi"
    elif prediksi_ipm >= 60:
        kategori = "Sedang"
    else:
        kategori = "Rendah"

    if selisih > 0:
        tren_text = "Meningkat"
    elif selisih < 0:
        tren_text = "Menurun"
    else:
        tren_text = "Stabil"

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Prediksi IPM",
        f"{prediksi_ipm:.2f}"
    )

    col2.metric(
        "Status IPM",
        kategori
    )

    col3.metric(
        "Tren",
        tren_text
    )

    st.markdown("---")

    st.subheader("Analisis Data")

    if tahun_pilihan == tahun_prediksi[0]:
        sumber_banding = "IPM prediksi tahun terakhir"
    else:
        sumber_banding = f"prediksi tahun {tahun_pilihan - 1}"

    st.info(
        f"""
        Pada tahun {tahun_pilihan},
        IPM Kota Magelang diprediksi mencapai
        {prediksi_ipm:.2f}.

        Status pembangunan manusia berada pada
        kategori {kategori}.

        Dibandingkan dengan {sumber_banding}
        sebesar {pembanding:.2f}, IPM diperkirakan
        {tren_text.lower()} sebesar
        {abs(selisih):.2f} poin.
        """
    )

elif menu == "Analisis Faktor":

    st.title("Analisis Faktor yang Mempengaruhi IPM")

    mean_shap = mean_shap.sort_values(
    by="Mean_SHAP",
    ascending=False
    ).reset_index(drop=True)

    fig, ax = plt.subplots(
        figsize=(5,2)
    )

    ax.barh(
            mean_shap["Feature"],
            mean_shap["Mean_SHAP"]
    )

    ax.invert_yaxis()

    ax.set_title(
    "SHAP Feature Importance"
    )
    st.pyplot(fig)

    mean_shap_display = mean_shap.rename(
    columns={
        "Feature": "Faktor Penentu IPM",
        "Mean_SHAP": "Nilai Pengaruh Rata-rata"
        }
    )

    top5 = mean_shap_display.head(5)

    st.subheader("Faktor Pendorong Utama")
    st.dataframe(top5, use_container_width=True)

    f1 = top5.iloc[0]["Faktor Penentu IPM"]
    f2 = top5.iloc[1]["Faktor Penentu IPM"]
    f3 = top5.iloc[2]["Faktor Penentu IPM"]
    f4 = top5.iloc[3]["Faktor Penentu IPM"]
    f5 = top5.iloc[4]["Faktor Penentu IPM"]

    st.subheader("Rekomendasi Kebijakan")

    sudah_tampil = set()

    for fitur in top5["Faktor Penentu IPM"]:

        if fitur in ["Bekerja", "Pengangguran"]:
            if "ketenagakerjaan" not in sudah_tampil:
                st.info(
                    "Meningkatkan kesempatan kerja dan menurunkan tingkat pengangguran "
                    "melalui penciptaan lapangan kerja, pelatihan tenaga kerja, "
                    "serta pengembangan UMKM."
                )
                sudah_tampil.add("ketenagakerjaan")

        elif fitur in ["BalitaGiziKurang", "BalitaGiziBaik"]:
            if "gizi" not in sudah_tampil:
                st.info(
                    "Memperkuat layanan kesehatan ibu dan anak, meningkatkan program "
                    "perbaikan gizi balita, serta edukasi gizi untuk meningkatkan "
                    "kualitas kesehatan masyarakat."
                )
                sudah_tampil.add("gizi")

        elif fitur == "Pertumbuhan_Ekonomi":
            if "ekonomi" not in sudah_tampil:
                st.info(
                    "Mendorong pertumbuhan ekonomi daerah melalui peningkatan investasi, "
                    "pengembangan UMKM, serta penciptaan lapangan kerja."
                )
                sudah_tampil.add("ekonomi")

        elif fitur in ["AngkaHarapanHidup", "AngkaKesakitan"]:
            if "kesehatan" not in sudah_tampil:
                st.info(
                    "Meningkatkan kualitas layanan kesehatan, memperluas akses fasilitas "
                    "kesehatan, serta memperkuat upaya promotif dan preventif."
                )
                sudah_tampil.add("kesehatan")

        elif fitur in ["Angka_Melek_Huruf", "APS", "Rata_Rata_Lama_Sekolah"]:
            if "pendidikan" not in sudah_tampil:
                st.info(
                    "Meningkatkan kualitas pendidikan melalui pemerataan akses pendidikan, "
                    "program beasiswa, peningkatan literasi, serta dukungan agar masyarakat "
                    "dapat menyelesaikan pendidikan hingga jenjang yang lebih tinggi."
                )
                sudah_tampil.add("pendidikan")

        elif fitur == "Jumlah_Penduduk":
            if "penduduk" not in sudah_tampil:
                st.info(
                    "Mengoptimalkan perencanaan pembangunan, penyediaan layanan publik, "
                    "serta pengendalian laju pertumbuhan penduduk agar pembangunan lebih merata."
                )
                sudah_tampil.add("penduduk")

    st.subheader("Faktor Yang Perlu Diperhatikan")

    st.info(
        f"""
        Faktor utama yang mempengaruhi IPM adalah
        {top5.iloc[0]['Faktor Penentu IPM']}, {top5.iloc[1]['Faktor Penentu IPM']},
        dan {top5.iloc[2]['Faktor Penentu IPM']}.

        Prioritas kebijakan sebaiknya difokuskan
        pada indikator-indikator tersebut untuk
        mendukung peningkatan IPM di masa mendatang.
        """)

elif menu == "Informasi Sistem":

    st.title("Informasi Performa Model Sistem")

    st.subheader("1. Evaluasi Model Prediksi")
    evaluasi_prophet = pd.read_excel(r"C:\SKRIPSI TUGAS AKHIR\result\evaluasi_prophet.xlsx")

    col_p1, col_p2, col_p3 = st.columns(3)

    col_p1.metric(
        label="MAE",
        value=evaluasi_prophet.loc[0, "MAE"]
    )

    col_p2.metric(
        label="RMSE",
        value=evaluasi_prophet.loc[0, "RMSE"]
    )

    col_p3.metric(
        label="MAPE",
        value=evaluasi_prophet.loc[0, "MAPE"]
    )
    st.markdown("---")

    st.subheader("2. Evaluasi Model Kontribusi Faktor")
    evaluasi_xgboost = pd.read_excel(r"C:\SKRIPSI TUGAS AKHIR\result\evaluasi_xgboost.xlsx")

    col_x1, col_x2, col_x3 = st.columns(3)

    col_x1.metric(
        label="MAE",
        value=evaluasi_xgboost.loc[0, "MAE"]
    )

    col_x2.metric(
        label="RMSE",
        value=evaluasi_xgboost.loc[0, "RMSE"]
    )

    col_x3.metric(
        label="R²",
        value=evaluasi_xgboost.loc[0, "R²"]
    )
    st.markdown("---")

    st.subheader("Interpretasi Kinerja Komponen")

    mae = evaluasi_prophet.loc[0, "MAE"]
    rmse = evaluasi_prophet.loc[0, "RMSE"]
    mape = evaluasi_prophet.loc[0, "MAPE"]

    xgb_mae_cv = evaluasi_xgboost.loc[0, "MAE"]
    xgb_rmse_cv = evaluasi_xgboost.loc[0, "RMSE"]
    xgb_r2_cv = evaluasi_xgboost.loc[0, "R²"]

    if mape < 10:
        kategori_mape = "Sangat Baik"
    elif mape < 20:
        kategori_mape = "Baik"
    elif mape < 50:
        kategori_mape = "Cukup"
    else:
        kategori_mape = "Kurang Baik"

    if xgb_r2_cv >= 0.90:
        kategori_r2 = "Sangat Kuat"
    elif xgb_r2_cv >= 0.70:
        kategori_r2 = "Kuat"
    elif xgb_r2_cv >= 0.50:
        kategori_r2 = "Sedang"
    else:
        kategori_r2 = "Lemah/Perlu Perbaikan"

    st.info(f"""
    ### Mean Absolute Error (MAE)

    Semakin kecil nilai MAE,
    semakin baik kemampuan model dalam
    melakukan prediksi.

    ---

    ### Root Mean Square Error (RMSE)

    Semakin kecil nilai RMSE,
    semakin baik performa model.

    ---

    ### Mean Absolute Percentage Error (MAPE)

    Interpretasi MAPE:

    • < 10% = Sangat Baik

    • 10% – 20% = Baik

    • 20% – 50% = Cukup

    • > 50% = Kurang Baik

    ---

    ### Koefisien Determinasi (R²)

    R² menunjukkan seberapa besar
    variasi IPM yang dapat dijelaskan
    oleh variabel dalam model.

    Semakin mendekati 1,
    semakin baik kemampuan model.
    """)

    st.subheader("Kesimpulan Evaluasi Sistem")

    if mape < 10 and xgb_r2_cv > 0.50:
        st.success(f"""
             * Akurasi prediksi IPM menghasilkan nilai kesalahan persentase (MAPE) sebesar **{mape:.2f}%** yang dikategorikan **{kategori_mape}**.
             * Analisis faktor IKM menghasilkan nilai $R^2$ sebesar **{xgb_r2_cv:.4f}** yang menduduki kategori tingkat penjelasan variansi **{kategori_r2}**.

            Model menunjukkan performa yang baik dalam melakukan prediksi IPM dan memberikan gambaran yang representatif terhadap variasi data."""
        )
    elif mape < 20 and xgb_r2_cv > 0.30:
        st.success(f"""
             * Akurasi prediksi IPM menghasilkan nilai kesalahan persentase (MAPE) sebesar **{mape:.2f}%** yang dikategorikan **{kategori_mape}**.
             * Analisis faktor IKM menghasilkan nilai $R^2$ sebesar **{xgb_r2_cv:.4f}** yang menduduki kategori tingkat penjelasan variansi **{kategori_r2}**.

            Model memiliki tingkat akurasi yang cukup baik dan dapat digunakan untuk membantu proses prediksi."""
        )
    else:
        st.warning(f"""
             * Akurasi prediksi IPM menghasilkan nilai kesalahan persentase (MAPE) sebesar **{mape:.2f}%** yang dikategorikan **{kategori_mape}**.
             * Analisis faktor IKM menghasilkan nilai $R^2$ sebesar **{xgb_r2_cv:.4f}** yang menduduki kategori tingkat penjelasan variansi **{kategori_r2}**.

            Model masih dapat digunakan, namun hasil prediksi perlu diinterpretasikan dengan hati-hati."""
        )

