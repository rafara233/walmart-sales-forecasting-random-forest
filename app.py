import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Prediksi Penjualan Walmart",
    layout="wide"
)

st.title("📊 Prediksi Penjualan Mingguan Walmart")

ATA_URL = "https://raw.githubusercontent.com/rafara233/walmart-sales-forecasting-random-forest/main/Walmart_Sales.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)

df = load_data()

st.header("📄 Dataset")

st.dataframe(df.head())

st.header("📊 Informasi Dataset")

col1, col2 = st.columns(2)

with col1:
    st.metric("Jumlah Data", f"{df.shape[0]:,}")
    st.metric("Jumlah Fitur", df.shape[1])
    st.metric("Jumlah Store", df["Store"].nunique())

with col2:
    st.metric("Missing Value", int(df.isnull().sum().sum()))
    st.metric("Data Duplikat", int(df.duplicated().sum()))
    st.metric("Rata-rata Weekly Sales", f"${df['Weekly_Sales'].mean():,.2f}")

st.subheader("Statistik Deskriptif")
st.dataframe(df.describe())

st.info("""
### 💡 Insight Dataset

- Dataset berisi data penjualan mingguan Walmart.
- Dataset memiliki **6.435 baris** dan **8 variabel** sebelum preprocessing.
- Tidak ditemukan **missing value**, sehingga data siap digunakan tanpa proses imputasi.
- Tidak ditemukan **data duplikat**, sehingga kualitas data tergolong baik.
- Target yang akan diprediksi adalah **Weekly_Sales**.
""")

    # ======================
    # Preprocessing
    # ======================

    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)

    df.drop("Date", axis=1, inplace=True)

    y = df["Weekly_Sales"]
    X = df.drop("Weekly_Sales", axis=1)

    # ======================
    # Perbandingan Split
    # ======================

    st.header("Perbandingan Rasio Train Test")

    split_ratio = {
        "90:10":0.10,
        "80:20":0.20,
        "70:30":0.30,
        "60:40":0.40
    }

    hasil=[]

    for rasio,test_size in split_ratio.items():

        X_train,X_test,y_train,y_test=train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42
        )

        model=RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train,y_train)

        pred=model.predict(X_test)

        hasil.append({
            "Split":rasio,
            "Train":len(X_train),
            "Test":len(X_test),
            "MAE":round(mean_absolute_error(y_test,pred),2),
            "RMSE":round(np.sqrt(mean_squared_error(y_test,pred)),2),
            "R2":round(r2_score(y_test,pred),4)
        })

    hasil_df=pd.DataFrame(hasil)

    st.dataframe(hasil_df)

    terbaik=hasil_df.loc[hasil_df["R2"].idxmax()]

    st.success(f"Split Terbaik : {terbaik['Split']}")

    # ======================
    # Training Model
    # ======================

    X_train,X_test,y_train,y_test=train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model=RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train,y_train)

    pred=model.predict(X_test)

    mae=mean_absolute_error(y_test,pred)
    rmse=np.sqrt(mean_squared_error(y_test,pred))
    r2=r2_score(y_test,pred)

    st.header("Evaluasi Model")

    c1,c2,c3=st.columns(3)

    c1.metric("MAE",f"{mae:,.2f}")
    c2.metric("RMSE",f"{rmse:,.2f}")
    c3.metric("R²",f"{r2:.4f}")

    # ======================
    # Feature Importance
    # ======================

    st.header("Feature Importance")

    importance=pd.DataFrame({
        "Feature":X.columns,
        "Importance":model.feature_importances_
    }).sort_values(
        by="Importance",
        ascending=False
    )

    st.dataframe(importance)

    st.bar_chart(
        importance.set_index("Feature")
    )

    # ======================
    # Prediksi
    # ======================

    st.header("Prediksi Penjualan")

    col1,col2,col3=st.columns(3)

    with col1:
        store=st.number_input("Store",1,50,5)
        holiday=st.selectbox(
            "Holiday Flag",
            [0,1]
        )
        temp=st.number_input(
            "Temperature",
            value=72.0
        )
        fuel=st.number_input(
            "Fuel Price",
            value=3.5
        )

    with col2:
        cpi=st.number_input(
            "CPI",
            value=210.0
        )
        unemployment=st.number_input(
            "Unemployment",
            value=6.0
        )
        year=st.number_input(
            "Year",
            value=2012
        )

    with col3:
        month=st.slider(
            "Month",
            1,
            12,
            10
        )

        week=st.slider(
            "Week",
            1,
            52,
            40
        )

    if st.button("Prediksi"):

        data_baru=pd.DataFrame({
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

        data_baru=data_baru[X.columns]

        hasil=model.predict(data_baru)

        st.success(
            f"Prediksi Penjualan Mingguan = ${hasil[0]:,.2f}"
        )
