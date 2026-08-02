import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt

# =====================================
# KONFIGURASI
# =====================================
st.set_page_config(
    page_title="Prediksi Penjualan Walmart",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Dashboard Prediksi Penjualan Walmart")
st.markdown("""
Aplikasi ini melakukan **Exploratory Data Analysis (EDA)** dan
**Prediksi Weekly Sales menggunakan Random Forest Regression**.
""")

# ==================================================
# GANTI DENGAN LINK RAW DATASET GITHUB
# ==================================================
GITHUB_RAW_URL = "https://raw.githubusercontent.com/username/repository/main/Walmart_Sales.csv"

# =====================================
# LOAD DATA
# =====================================
@st.cache_data
def load_data():
    df = pd.read_csv(GITHUB_RAW_URL)
    return df

df = load_data()

# =====================================
# EDA
# =====================================
st.header("1. Exploratory Data Analysis (EDA)")

tab1, tab2, tab3 = st.tabs([
    "Data",
    "Statistik",
    "Visualisasi"
])

with tab1:

    st.subheader("Preview Dataset")
    st.dataframe(df.head())

    st.subheader("Informasi Dataset")

    info = pd.DataFrame({
        "Kolom": df.columns,
        "Tipe Data": df.dtypes.astype(str),
        "Missing Value": df.isnull().sum().values
    })

    st.dataframe(info)

    st.metric("Jumlah Duplikat", df.duplicated().sum())

    st.info("""
    **Insight**
    - Dataset digunakan sebagai dasar pembangunan model prediksi.
    - Missing value dan data duplikat perlu diperiksa sebelum modeling.
    """)

with tab2:

    st.subheader("Statistik Deskriptif")
    st.dataframe(df.describe())

    st.info("""
    **Insight**
    - Statistik deskriptif membantu mengetahui persebaran nilai.
    - Weekly Sales memiliki rentang nilai yang cukup besar sehingga variasi penjualan antar toko cukup tinggi.
    """)

# =====================================
# PREPROCESSING
# =====================================
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Week"] = df["Date"].dt.isocalendar().week.astype(int)

# Visualisasi
with tab3:

    st.subheader("Distribusi Weekly Sales")

    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(df["Weekly_Sales"], bins=30)
    ax.set_xlabel("Weekly Sales")
    ax.set_ylabel("Jumlah")
    st.pyplot(fig)

    avg_sales = df.groupby("Store")["Weekly_Sales"].mean()

    st.subheader("Rata-rata Penjualan per Store")

    fig2, ax2 = plt.subplots(figsize=(10,5))
    avg_sales.plot(kind="bar", ax=ax2)

    st.pyplot(fig2)

    st.info("""
    **Insight**
    - Beberapa store memiliki rata-rata penjualan jauh lebih tinggi.
    - Distribusi penjualan tidak merata sehingga model Machine Learning diperlukan.
    """)

# =====================================
# MODELING
# =====================================

st.header("2. Machine Learning")

df_model = df.drop("Date", axis=1)

X = df_model.drop("Weekly_Sales", axis=1)
y = df_model["Weekly_Sales"]

split_ratio = {
    "90:10":0.10,
    "80:20":0.20,
    "70:30":0.30,
    "60:40":0.40
}

hasil = []

for nama,test_size in split_ratio.items():

    X_train,X_test,y_train,y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train,y_train)

    pred = model.predict(X_test)

    hasil.append({
        "Split":nama,
        "MAE":mean_absolute_error(y_test,pred),
        "RMSE":np.sqrt(mean_squared_error(y_test,pred)),
        "R2":r2_score(y_test,pred)
    })

hasil_df = pd.DataFrame(hasil)

st.subheader("Perbandingan Split Data")

st.dataframe(hasil_df)

best = hasil_df.loc[hasil_df["R2"].idxmax()]

st.success(f"Split terbaik berdasarkan R² adalah **{best['Split']}**")

# =====================================
# FINAL MODEL
# =====================================

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train,y_train)

pred = model.predict(X_test)

mae = mean_absolute_error(y_test,pred)
rmse = np.sqrt(mean_squared_error(y_test,pred))
r2 = r2_score(y_test,pred)

st.subheader("Evaluasi Model")

col1,col2,col3 = st.columns(3)

col1.metric("MAE",f"{mae:,.2f}")
col2.metric("RMSE",f"{rmse:,.2f}")
col3.metric("R²",f"{r2:.4f}")

st.info("""
### Insight Model
- Semakin kecil MAE dan RMSE maka prediksi semakin akurat.
- Nilai R² yang mendekati 1 menunjukkan model mampu menjelaskan variasi data dengan baik.
""")

# =====================================
# FEATURE IMPORTANCE
# =====================================

importance = pd.DataFrame({
    "Feature":X.columns,
    "Importance":model.feature_importances_
}).sort_values("Importance",ascending=False)

st.subheader("Feature Importance")

fig3,ax3 = plt.subplots(figsize=(8,5))

ax3.barh(
    importance["Feature"],
    importance["Importance"]
)

ax3.invert_yaxis()

st.pyplot(fig3)

st.info("""
### Insight
Fitur dengan nilai importance terbesar merupakan faktor yang paling memengaruhi prediksi Weekly Sales.
""")

# =====================================
# PREDIKSI
# =====================================

st.header("3. Prediksi Penjualan")

col1,col2,col3 = st.columns(3)

store = col1.number_input("Store",1,45,5)
holiday = col2.selectbox("Holiday Flag",[0,1])
temp = col3.number_input("Temperature",70.0)

fuel = col1.number_input("Fuel Price",3.5)
cpi = col2.number_input("CPI",210.0)
unemployment = col3.number_input("Unemployment",6.0)

year = col1.number_input("Year",2010,2025,2012)
month = col2.number_input("Month",1,12,10)
week = col3.number_input("Week",1,53,40)

if st.button("Prediksi"):

    data_baru = pd.DataFrame({

        "Store":[store],
        "Holiday_Flag":[holiday],
        "Temperature":[temp],
        "Fuel_Price":[fuel],
        "CPI":[cpi],
        "Unemployment":[unemployment],
        "Year":[year],
        "Month":[month],
        "Week":[week]

    })

    prediksi = model.predict(data_baru)

    st.success(
        f"Estimasi Weekly Sales = **${prediksi[0]:,.2f}**"
    )

# =====================================
# KESIMPULAN
# =====================================

st.header("4. Kesimpulan")

st.markdown("""
### Insight Akhir

- Dataset Walmart digunakan untuk memprediksi penjualan mingguan (Weekly Sales).
- Random Forest Regression dipilih karena mampu menangani hubungan non-linear dan menghasilkan performa yang baik.
- Evaluasi model menggunakan MAE, RMSE, dan R².
- Feature Importance menunjukkan faktor-faktor yang paling memengaruhi penjualan.
- Dashboard ini dapat membantu perusahaan memperkirakan penjualan berdasarkan kondisi ekonomi, musim, dan karakteristik toko.
""")
