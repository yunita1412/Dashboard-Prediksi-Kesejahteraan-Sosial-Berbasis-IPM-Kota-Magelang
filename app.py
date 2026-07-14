import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import plotly.express as px
import os

from sklearn.model_selection import KFold
from prophet import Prophet
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

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
    BASE_DIR,
    "dataset_final.xlsx"
)

MEAN_SHAP_PATH = os.path.join(
    BASE_DIR,
    "mean_shap.xlsx"
)

FEATURE_IMPORTANCE_PATH = os.path.join(
    BASE_DIR,
    "feature_importance.xlsx"
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

prophet_df = df[["ds","IPM"]].copy()
prophet_df.columns = ["ds","y"]
prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

split = int(len(prophet_df)*0.8)
train = prophet_df[:split]
test = prophet_df[split:]

prophet_model = Prophet(yearly_seasonality=True)
prophet_model.fit(train)

forecast_test = prophet_model.predict(test[["ds"]])

mae = mean_absolute_error(test["y"], forecast_test["yhat"])
rmse = np.sqrt(mean_squared_error(test["y"], forecast_test["yhat"]))
mape = mean_absolute_percentage_error(test["y"], forecast_test["yhat"]) * 100

future = prophet_model.make_future_dataframe(
    periods=30, 
    freq="QS"
)

forecast = prophet_model.predict(future)

tahun_aktual_terakhir = int(df["Tahun"].max())

tahun_prediksi = sorted(
    forecast[
        forecast["ds"].dt.year > tahun_aktual_terakhir
    ]["ds"].dt.year.unique()
)

target = "IPM"

X_all = df.drop(columns=["IPM", "Tahun", "Triwulan", "ds", "TW_num"], errors="ignore")
X_all = X_all.select_dtypes(include=["int64", "float64"])
y = df[target]

xgb_baseline = xgb.XGBRegressor(
    n_estimators=50, max_depth=2, random_state=42
)
xgb_baseline.fit(X_all, y)

baseline_importance = pd.DataFrame({
    "Fitur": X_all.columns,
    "Importance": xgb_baseline.feature_importances_
}).sort_values(by="Importance", ascending=False)

top_5_features = baseline_importance["Fitur"].head(5).tolist()

X_selected = X_all[top_5_features].copy()

kf = KFold(n_splits=5, shuffle=True, random_state=42)
xgb_mae_scores, xgb_rmse_scores, xgb_r2_scores = [], [], []

for train_idx, test_idx in kf.split(X_selected, y):
    X_train_f, X_test_f = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
    y_train_f, y_test_f = y.iloc[train_idx], y.iloc[test_idx]
    
    fold_model = xgb.XGBRegressor(
        n_estimators=60, learning_rate=0.03, max_depth=2,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    fold_model.fit(X_train_f, y_train_f)
    preds_f = fold_model.predict(X_test_f)
    
    xgb_mae_scores.append(mean_absolute_error(y_test_f, preds_f))
    xgb_rmse_scores.append(np.sqrt(mean_squared_error(y_test_f, preds_f)))
    xgb_r2_scores.append(r2_score(y_test_f, preds_f))

xgb_mae_cv = np.mean(xgb_mae_scores)
xgb_rmse_cv = np.mean(xgb_rmse_scores)
xgb_r2_cv = np.mean(xgb_r2_scores)

xgb_model_final = xgb.XGBRegressor(
    n_estimators=60, learning_rate=0.03, max_depth=2,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
xgb_model_final.fit(X_selected, y)
xgb_model_final.save_model("model_xgb_ipm.json")

mean_shap = pd.read_excel(
    MEAN_SHAP_PATH
)

feature_importance = pd.read_excel(
    FEATURE_IMPORTANCE_PATH
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

    tahun_pilihan = st.selectbox(
        "Pilih Tahun Prediksi",
        tahun_prediksi
    )

    forecast_tahun = forecast[
        forecast["ds"].dt.year == tahun_pilihan
    ]

    prediksi_ipm = forecast_tahun["yhat"].mean()

    forecast_plot = forecast.copy()
    forecast_plot["Tahun"] = forecast_plot["ds"].dt.year

    pred_tahun = (
        forecast_plot.groupby("Tahun")["yhat"]
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
        forecast_plot.groupby("Tahun")["yhat"]
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
    
    fig, ax = plt.subplots(
        figsize=(5,2)
    )

    ax.barh(
            mean_shap["Feature"],
            mean_shap["Mean_SHAP"]
    )

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
    
    rekomendasi = {
        "Bekerja": "Meningkatkan kesempatan kerja dan produktivitas tenaga kerja.",
        "Pengangguran": "Menurunkan tingkat pengangguran melalui penciptaan lapangan kerja.",
        "GiziKurang": "Meningkatkan program perbaikan gizi masyarakat.",
        "GiziBaik": "Memperkuat layanan Posyandu dan pemantauan kesehatan balita.",
        "Pertumbuhan_Ekonomi": "Mendorong pertumbuhan ekonomi daerah melalui investasi dan UMKM."
    }
    
    for fitur in top5["Faktor Penentu IPM"]:
        if fitur in rekomendasi:
            st.info(rekomendasi[fitur])
    
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

    col_p1, col_p2, col_p3 = st.columns(3)

    col_p1.metric("MAE", f"{mae:.4f}")
    col_p2.metric("RMSE", f"{rmse:.4f}")
    col_p3.metric("MAPE", f"{mape:.2f}%")

    st.markdown("---")

    st.subheader("2. Evaluasi Model Kontribusi Faktor")
    col_x1, col_x2, col_x3 = st.columns(3)
    
    col_x1.metric("MAE", round(xgb_mae_cv, 4))
    col_x2.metric("RMSE", round(xgb_rmse_cv, 4))
    col_x3.metric("R² Score", round(xgb_r2_cv, 4))

    st.markdown("---")
    st.subheader("Interpretasi Kinerja Komponen")

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
