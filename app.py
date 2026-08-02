import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from streamlit_option_menu import option_menu

import matplotlib.pyplot as plt

# ==========================================================
# STYLE TABLE
# ==========================================================

def style_table(df):

    numeric_cols = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    text_cols = df.select_dtypes(
        exclude=["int64", "float64"]
    ).columns


    styler = df.style.set_table_styles([

        # Header kolom
        {
            "selector": "th",
            "props": [
                ("text-align", "center"),
                ("font-weight", "bold")
            ]
        },

        # Semua isi tabel
        {
            "selector": "td",
            "props": [
                ("font-size", "14px")
            ]
        }

    ])


    # Kolom angka rata tengah
    if len(numeric_cols) > 0:

        styler = styler.set_properties(
            subset=numeric_cols,
            **{
                "text-align": "center !important"
            }
        )


    # Kolom teks rata kiri
    if len(text_cols) > 0:

        styler = styler.set_properties(
            subset=text_cols,
            **{
                "text-align": "left !important"
            }
        )


    return styler

# ==========================================================
# FUNGSI FORMAT TABEL
# ==========================================================

def show_table(df):

    config = {}

    for col in df.columns:

        if pd.api.types.is_numeric_dtype(df[col]):

            config[col] = st.column_config.NumberColumn(
                label=col,
                format="%.2f",
                width="medium"
            )

        else:

            config[col] = st.column_config.TextColumn(
                label=col,
                width="medium"
            )


    st.dataframe(
        df,
        column_config=config,
        use_container_width=True,
        hide_index=True
    )

# ==========================================================
# KONFIGURASI HALAMAN
# ==========================================================

st.set_page_config(
    page_title="Dashboard Prediksi Penjualan Walmart",
    #page_icon="📈",
    layout="wide"
)
st.markdown("""
<style>

div.stButton > button {
    color: black !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    background-color: white !important;
    border: none !important;
    height: 50px;
    border-radius: 8px;
}

/* Saat mouse diarahkan */
div.stButton > button:hover {
    color: black !important;
    background-color: #eeeeee !important;
}

/* Semua teks pada menu */
div.stButton > button p {
    color: black !important;
    font-weight: 700 !important;
}

</style>
""", unsafe_allow_html=True)

st.title("Dashboard Prediksi Penjualan Walmart")

st.caption(
    "Analisis Dataset Walmart dan Prediksi Weekly Sales menggunakan Random Forest Regression."
)
menu = option_menu(
    menu_title=None,
    options=[
        "Dataset",
        "Statistik",
        "Machine Learning",
        "Prediksi",
        "Kesimpulan"
    ],
    icons=[
        "table",
        "bar-chart",
        "cpu",
        "graph-up-arrow",
        "clipboard-check"
    ],
    orientation="horizontal",
    default_index=0,

    styles={

        # Background keseluruhan menu
        "container": {
            "padding": "0!important",
            "background-color": "white!important"
        },

        # Icon
        "icon": {
            "color": "black!important",
            "font-size": "20px"
        },

        # Tulisan menu tidak aktif
        "nav-link": {
            "font-size": "18px",
            "font-weight": "700",
            "color": "black!important",
            "background-color": "white!important",
            "text-align": "center",
            "margin": "0px",
            "padding": "12px"
        },

        # Menu aktif
        "nav-link-selected": {
            "background-color": "#e5e5e5!important",
            "color": "black!important",
            "font-weight": "800"
        }

    }
)

# ==========================================================
# LOAD DATASET
# ==========================================================

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "rafara233/"
    "walmart-sales-forecasting-random-forest/"
    "refs/heads/main/"
    "Walmart_Sales.csv"
)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    return df

df = load_data()

# ==========================================================
# PREPROCESSING
# ==========================================================

df["Date"] = pd.to_datetime(
    df["Date"],
    format="%d-%m-%Y"
)

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Week"] = (
    df["Date"]
    .dt
    .isocalendar()
    .week
    .astype(int)
)

# ==========================================================
# DATA MODEL
# ==========================================================

df_model = df.drop(columns=["Date"])

X = df_model.drop(columns=["Weekly_Sales"])
y = df_model["Weekly_Sales"]

# ==========================================================
# PERBANDINGAN SPLIT DATA
# ==========================================================

split_ratio = {
    "90 : 10": 0.10,
    "80 : 20": 0.20,
    "70 : 30": 0.30,
    "60 : 40": 0.40
}

hasil_split = []

for nama, test_size in split_ratio.items():

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42
    )

    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    rf.fit(X_train, y_train)

    pred = rf.predict(X_test)

    hasil_split.append({

        "Split": nama,

        "MAE":
        mean_absolute_error(
            y_test,
            pred
        ),

        "RMSE":
        np.sqrt(
            mean_squared_error(
                y_test,
                pred
            )
        ),

        "R²":
        r2_score(
            y_test,
            pred
        )

    })

hasil_split = pd.DataFrame(hasil_split)

best_split = hasil_split.loc[
    hasil_split["R²"].idxmax()
]

# ==========================================================
# MODEL TERBAIK
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        pred
    )
)

r2 = r2_score(
    y_test,
    pred
)

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance":
    model.feature_importances_

}).sort_values(
    by="Importance",
    ascending=False
)

# ==========================================================
# KORELASI
# ==========================================================

corr = df_model.corr(
    numeric_only=True
)

# ==========================================================
# MENU DATASET
# ==========================================================

if menu == "Dataset":

    st.header("Dataset Walmart")

    st.markdown("""
    Menu ini menampilkan informasi umum mengenai dataset yang digunakan
    sebagai dasar pembangunan model Machine Learning.
    """)

    # ======================================================
    # METRIC
    # ======================================================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Jumlah Baris",
        f"{df.shape[0]:,}"
    )

    col2.metric(
        "Jumlah Kolom",
        df.shape[1]
    )

    col3.metric(
        "Missing Value",
        int(df.isnull().sum().sum())
    )

    col4.metric(
        "Data Duplikat",
        int(df.duplicated().sum())
    )

    st.divider()

    # ======================================================
    # PREVIEW DATASET
    # ======================================================

    st.subheader("Preview Dataset")

    jumlah = st.slider(
        "Jumlah data yang ditampilkan",
        min_value=5,
        max_value=30,
        value=10
    )

    st.dataframe(
        show_table(df.head(jumlah)),
        use_container_width=True
    )

    st.divider()

    # ======================================================
    # INFORMASI DATASET
    # ======================================================

    st.subheader("Informasi Dataset")

    info = pd.DataFrame({

        "Nama Kolom": df.columns,

        "Tipe Data": df.dtypes.astype(str),

        "Missing Value": df.isnull().sum().values,

        "Jumlah Nilai Unik":
        [df[col].nunique() for col in df.columns]

    })

    st.dataframe(
        show_table(info),
        use_container_width=True
    )

    st.divider()

    # ======================================================
    # DESKRIPSI KOLOM
    # ======================================================

    st.subheader("Deskripsi Setiap Kolom")

    deskripsi = pd.DataFrame({

        "Kolom":[
            "Store",
            "Date",
            "Holiday_Flag",
            "Temperature",
            "Fuel_Price",
            "CPI",
            "Unemployment",
            "Weekly_Sales",
            "Year",
            "Month",
            "Week"
        ],

        "Keterangan":[
            "Nomor toko Walmart.",
            "Tanggal penjualan.",
            "0 = Bukan hari libur, 1 = Hari libur.",
            "Suhu rata-rata.",
            "Harga bahan bakar.",
            "Consumer Price Index.",
            "Tingkat pengangguran.",
            "Total penjualan mingguan.",
            "Tahun transaksi.",
            "Bulan transaksi.",
            "Minggu transaksi."
        ]

    })

    st.dataframe(
        show_table(deskripsi),
        use_container_width=True
    )

    st.divider()

    # ======================================================
    # RENTANG DATASET
    # ======================================================

    st.subheader("Rentang Dataset")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Tahun Awal",
        int(df["Year"].min())
    )

    col2.metric(
        "Tahun Akhir",
        int(df["Year"].max())
    )

    col3.metric(
        "Jumlah Store",
        df["Store"].nunique()
    )

    st.info("""
Dataset Walmart yang digunakan pada penelitian ini
mencakup data penjualan dari **tahun 2011 hingga 2012**.
Walaupun pada menu prediksi pengguna dapat memasukkan tahun
di atas 2012, model tetap mempelajari pola berdasarkan
dataset periode 2011–2012.
""")

    st.divider()

    # ======================================================
    # NILAI HILANG
    # ======================================================

    st.subheader("Pemeriksaan Missing Value")

    missing = pd.DataFrame({

        "Kolom":df.columns,

        "Missing Value":
        df.isnull().sum().values

    })

    st.dataframe(
       show_table(missing.style.set_properties(
          **{
             "text-align": "left"
            }
        )),
        use_container_width=True
    )

    if df.isnull().sum().sum() == 0:

        st.success(
            "Tidak ditemukan missing value pada dataset."
        )

    else:

        st.warning(
            "Masih terdapat missing value."
        )

    st.divider()

    # ======================================================
    # DUPLIKAT
    # ======================================================

    st.subheader("Pemeriksaan Data Duplikat")

    duplicate = df.duplicated().sum()

    st.metric(
        "Jumlah Data Duplikat",
        duplicate
    )

    if duplicate == 0:

        st.success(
            "Tidak ditemukan data duplikat."
        )

    else:

        st.warning(
            "Terdapat data duplikat yang sebaiknya dibersihkan."
        )

    st.divider()

    # ======================================================
    # INSIGHT
    # ======================================================

    st.subheader("Insight Dataset")

    st.success(f"""
1. Dataset terdiri dari **{df.shape[0]:,} baris**
   dan **{df.shape[1]} kolom**.

2. Dataset mencakup data penjualan Walmart
   selama periode **2011–2012**.

3. Dataset digunakan sebagai dasar
   pembangunan model Random Forest Regression.

4. Tidak ditemukan missing value sehingga
   data siap digunakan untuk proses Machine Learning.

5. Pemeriksaan data duplikat membantu memastikan
   kualitas data sebelum proses pelatihan model.

6. Variabel target yang diprediksi adalah
   **Weekly Sales (USD)**.

7. Variabel Store, Holiday Flag, Temperature,
   Fuel Price, CPI, Unemployment, Year,
   Month, dan Week digunakan sebagai fitur
   untuk memprediksi Weekly Sales.
""")

# ==========================================================
# MENU STATISTIK
# ==========================================================

elif menu == "Statistik":

    st.header("📊 Statistik Dataset")

    st.markdown("""
    Menu ini menyajikan analisis statistik deskriptif untuk memahami karakteristik data
    sebelum dilakukan proses Machine Learning.
    """)

    # ======================================================
    # STATISTIK DESKRIPTIF
    # ======================================================

    st.subheader("Statistik Deskriptif")

    statistik = df.describe().T

    statistik = statistik.rename(columns={
        "count": "Jumlah Data",
        "mean": "Rata-rata",
        "std": "Standar Deviasi",
        "min": "Minimum",
        "25%": "Kuartil 1",
        "50%": "Median",
        "75%": "Kuartil 3",
        "max": "Maksimum"
    })

    st.dataframe(
        show_table(statistik),
        use_container_width=True
    )

    st.info("""
**Insight Statistik Deskriptif**

- Nilai rata-rata menunjukkan kecenderungan pusat dari setiap variabel.
- Standar deviasi menunjukkan tingkat penyebaran data.
- Semakin besar standar deviasi, semakin besar variasi data.
- Nilai minimum dan maksimum menunjukkan rentang data yang dimiliki masing-masing variabel.
""")

    st.divider()

    # ======================================================
    # HISTOGRAM WEEKLY SALES
    # ======================================================

    st.subheader("Distribusi Weekly Sales")

    fig, ax = plt.subplots(figsize=(10,5))

    ax.hist(
        df["Weekly_Sales"],
        bins=30,
        edgecolor="black"
    )

    ax.set_title("Histogram Weekly Sales")

    ax.set_xlabel("Weekly Sales (USD)")

    ax.set_ylabel("Frekuensi")

    st.pyplot(fig)

    st.info("""
**Insight Histogram**

- Histogram menunjukkan persebaran nilai Weekly Sales.
- Distribusi yang tidak merata menandakan adanya variasi penjualan antar toko.
- Nilai penjualan yang tinggi maupun rendah tetap dipelajari oleh model Random Forest.
""")

    st.divider()

    # ======================================================
    # BOXPLOT
    # ======================================================

    st.subheader("Boxplot Weekly Sales")

    fig2, ax2 = plt.subplots(figsize=(10,3))

    ax2.boxplot(df["Weekly_Sales"], vert=False)

    ax2.set_xlabel("Weekly Sales (USD)")

    st.pyplot(fig2)

    st.info("""
**Insight Boxplot**

- Boxplot digunakan untuk mendeteksi adanya outlier.
- Titik yang berada di luar whisker menunjukkan nilai yang jauh berbeda dibandingkan sebagian besar data.
- Random Forest relatif tahan terhadap keberadaan outlier sehingga model masih dapat bekerja dengan baik.
""")

    st.divider()

    # ======================================================
    # RATA-RATA PENJUALAN PER STORE
    # ======================================================

    st.subheader("Rata-rata Weekly Sales per Store")

    avg_store = (
        df.groupby("Store")["Weekly_Sales"]
        .mean()
        .sort_index()
    )

    fig3, ax3 = plt.subplots(figsize=(12,5))

    ax3.bar(
        avg_store.index.astype(str),
        avg_store.values
    )

    ax3.set_title("Rata-rata Weekly Sales per Store")
    ax3.set_xlabel("Store")
    ax3.set_ylabel("Weekly Sales (USD)")

    plt.xticks(rotation=90)

    st.pyplot(fig3)

    st.info(f"""
**Insight**

- Store dengan rata-rata penjualan tertinggi adalah **Store {avg_store.idxmax()}**
  dengan rata-rata sekitar **USD ${avg_store.max():,.2f}**.

- Store dengan rata-rata penjualan terendah adalah **Store {avg_store.idxmin()}**
  dengan rata-rata sekitar **USD ${avg_store.min():,.2f}**.

- Perbedaan rata-rata penjualan antar store menunjukkan bahwa setiap toko memiliki karakteristik penjualan yang berbeda.
""")

    st.divider()

    # ======================================================
    # RATA-RATA PENJUALAN PER BULAN
    # ======================================================

    st.subheader("Rata-rata Weekly Sales per Bulan")

    avg_month = (
        df.groupby("Month")["Weekly_Sales"]
        .mean()
        .sort_index()
    )

    fig4, ax4 = plt.subplots(figsize=(10,4))

    ax4.plot(
        avg_month.index,
        avg_month.values,
        marker="o"
    )

    ax4.set_xticks(range(1,13))

    ax4.set_xlabel("Month")
    ax4.set_ylabel("Weekly Sales (USD)")
    ax4.set_title("Rata-rata Weekly Sales per Bulan")

    st.pyplot(fig4)

    st.info("""
**Insight**

- Grafik ini menunjukkan perubahan rata-rata penjualan pada setiap bulan.
- Bulan dengan rata-rata penjualan tinggi menunjukkan periode dengan aktivitas penjualan yang lebih besar.
- Informasi ini membantu memahami pola musiman (seasonality) pada data.
""")

    st.divider()

    # ======================================================
    # RINGKASAN STATISTIK
    # ======================================================

    st.subheader("Ringkasan Statistik Weekly Sales")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Rata-rata",
        f"USD ${df['Weekly_Sales'].mean():,.2f}"
    )

    col2.metric(
        "Median",
        f"USD ${df['Weekly_Sales'].median():,.2f}"
    )

    col3.metric(
        "Standar Deviasi",
        f"USD ${df['Weekly_Sales'].std():,.2f}"
    )

    col4, col5 = st.columns(2)

    col4.metric(
        "Minimum",
        f"USD ${df['Weekly_Sales'].min():,.2f}"
    )

    col5.metric(
        "Maksimum",
        f"USD ${df['Weekly_Sales'].max():,.2f}"
    )

    st.divider()

    # ======================================================
    # KESIMPULAN STATISTIK
    # ======================================================

    st.subheader("Kesimpulan Analisis Statistik")

    st.success(f"""
1. Dataset memiliki **{len(df):,}** data yang siap digunakan untuk proses Machine Learning.

2. Nilai rata-rata Weekly Sales adalah sekitar **USD ${df['Weekly_Sales'].mean():,.2f}**.

3. Nilai maksimum Weekly Sales mencapai **USD ${df['Weekly_Sales'].max():,.2f}**, sedangkan nilai minimum sebesar **USD ${df['Weekly_Sales'].min():,.2f}**.

4. Terdapat perbedaan rata-rata penjualan antar store, yang menunjukkan bahwa karakteristik setiap toko tidak sama.

5. Pola penjualan bulanan menunjukkan adanya perubahan rata-rata penjualan pada periode tertentu sehingga faktor waktu menjadi salah satu variabel penting.

6. Berdasarkan hasil analisis statistik, variabel-variabel pada dataset layak digunakan untuk membangun model prediksi menggunakan Random Forest Regression.
""")

# ==========================================================
# MENU MACHINE LEARNING
# ==========================================================

elif menu == "Machine Learning":

    st.header("🤖 Machine Learning - Random Forest Regression")

    st.markdown("""
Pada penelitian ini digunakan algoritma **Random Forest Regression**
untuk memprediksi nilai **Weekly Sales (USD)**.

Random Forest merupakan algoritma Machine Learning bertipe **Supervised Learning**
yang bekerja dengan membangun banyak Decision Tree, kemudian
menggabungkan hasil prediksi seluruh tree sehingga menghasilkan prediksi
yang lebih stabil dan akurat.
""")

    # ======================================================
    # MENGAPA RANDOM FOREST
    # ======================================================

    st.subheader("Mengapa Menggunakan Random Forest Regression?")

    alasan = pd.DataFrame({

        "Alasan":[
            "Mampu menangani hubungan non-linear",
            "Tidak mudah mengalami overfitting",
            "Mampu menangani banyak variabel",
            "Tidak memerlukan normalisasi data",
            "Dapat menghitung Feature Importance",
            "Performa prediksi umumnya tinggi"
        ],

        "Keterangan":[
            "Hubungan antar variabel tidak harus berbentuk garis lurus.",
            "Menggabungkan banyak Decision Tree sehingga lebih stabil.",
            "Cocok untuk dataset dengan banyak fitur.",
            "Data asli dapat langsung digunakan.",
            "Mengetahui variabel paling berpengaruh.",
            "Sering memberikan hasil yang baik pada data regresi."
        ]

    })

    st.dataframe(
        alasan,
        use_container_width=True
    )

    st.success("""
### Kesimpulan

Random Forest Regression dipilih karena mampu menghasilkan prediksi
yang akurat, stabil, serta dapat menunjukkan variabel mana yang
paling memengaruhi nilai Weekly Sales.
""")

    st.divider()

    # ======================================================
    # FITUR DAN TARGET
    # ======================================================

    st.subheader("Fitur dan Target")

    col1, col2 = st.columns(2)

    with col1:

        st.write("### Variabel Input (Feature)")

        fitur = pd.DataFrame({
            "Feature": X.columns
        })

        st.dataframe(
            style_table(fitur),
            use_container_width=True,
            hide_index=True
        )

    with col2:

        st.write("### Variabel Target")

        target = pd.DataFrame({
            "Target":[
                "Weekly_Sales"
            ],
            "Keterangan":[
                "Total penjualan mingguan (USD)"
            ]
        })

        st.dataframe(
            style_table(target),
            use_container_width=True,
            hide_index=True
        )

    st.info("""
Model akan mempelajari hubungan antara seluruh variabel input
(Store, Holiday Flag, Temperature, Fuel Price, CPI, Unemployment,
Year, Month, dan Week) terhadap nilai Weekly Sales.
""")

    st.divider()

    # ======================================================
    # PERBANDINGAN SPLIT DATA
    # ======================================================

    st.subheader("Perbandingan Split Data")

    st.dataframe(
        hasil_split.style.format({
            "MAE":"{:.2f}",
            "RMSE":"{:.2f}",
            "R²":"{:.4f}"
        }),
        use_container_width=True
    )

    st.success(
        f"Split terbaik berdasarkan nilai R² adalah **{best_split['Split']}**."
    )

    st.divider()

    # ======================================================
    # PENJELASAN METRIK
    # ======================================================

    st.subheader("Penjelasan Metrik Evaluasi")

    metrik = pd.DataFrame({

        "Metrik":[
            "MAE",
            "RMSE",
            "R² Score"
        ],

        "Fungsi":[
            "Mengukur rata-rata kesalahan prediksi.",
            "Mengukur besar kesalahan dengan memberi penalti lebih besar pada error yang tinggi.",
            "Mengukur kemampuan model menjelaskan variasi data."
        ],

        "Interpretasi":[
            "Semakin kecil semakin baik.",
            "Semakin kecil semakin baik.",
            "Semakin mendekati 1 semakin baik."
        ]

    })

    st.dataframe(
        style_table(metrik),
        use_container_width=True,
        hide_index=True
    )

    st.info("""
### Insight

- Nilai MAE dan RMSE digunakan untuk mengetahui besar kesalahan prediksi.
- Nilai R² menunjukkan seberapa baik model mampu menjelaskan variasi Weekly Sales.
- Split data dengan nilai R² tertinggi dipilih sebagai acuan karena memberikan performa terbaik pada data uji.
""")

    st.divider()

    # ======================================================
    # KORELASI ANTAR KOLOM
    # ======================================================

    st.subheader("Korelasi Antar Variabel")

    st.markdown("""
Korelasi digunakan untuk mengetahui hubungan antar variabel numerik.

Nilai korelasi berada pada rentang **-1 sampai 1**.

- Nilai mendekati **1**  → hubungan positif sangat kuat.
- Nilai mendekati **-1** → hubungan negatif sangat kuat.
- Nilai mendekati **0**  → hampir tidak memiliki hubungan.
""")

    # ==========================================
    # HEATMAP
    # ==========================================

    fig, ax = plt.subplots(figsize=(10,8))

    im = ax.imshow(
        corr,
        cmap="coolwarm"
    )

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(
        corr.columns,
        rotation=90
    )

    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)

    plt.colorbar(im)

    st.pyplot(fig)

    st.divider()

    # ==========================================
    # TABEL KORELASI
    # ==========================================

    st.subheader("Tabel Korelasi")

    st.dataframe(
        corr.round(3),
        use_container_width=True
    )

    st.divider()

    # ==========================================
    # KORELASI TERHADAP TARGET
    # ==========================================

    st.subheader("Korelasi Terhadap Weekly Sales")

    corr_target = (
        corr["Weekly_Sales"]
        .drop("Weekly_Sales")
        .sort_values(
            ascending=False
        )
    )

    corr_df = pd.DataFrame({

        "Feature": corr_target.index,

        "Correlation":
        corr_target.values

    })

    st.dataframe(
        style_table(corr_df),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ==========================================
    # INTERPRETASI
    # ==========================================

    st.subheader("Interpretasi Tingkat Korelasi")

    interpretasi = pd.DataFrame({

        "Nilai Korelasi":[
            "0.80 – 1.00",
            "0.60 – 0.79",
            "0.40 – 0.59",
            "0.20 – 0.39",
            "0.00 – 0.19"
        ],

        "Kategori":[
            "Sangat Kuat",
            "Kuat",
            "Sedang",
            "Lemah",
            "Sangat Lemah"
        ]

    })

    st.dataframe(
        style_table(interpretasi),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ==========================================
    # VARIABEL PALING BERPENGARUH
    # ==========================================

    st.subheader("Analisis Hubungan Variabel")

    terbesar = corr_target.abs().idxmax()
    nilai = corr_target[terbesar]

    st.success(f"""
Variabel yang memiliki hubungan paling kuat terhadap **Weekly Sales**
berdasarkan analisis korelasi adalah **{terbesar}**
dengan nilai korelasi **{nilai:.3f}**.
""")

    st.info(f"""
### Insight

1. Variabel **{terbesar}** merupakan variabel yang memiliki hubungan paling besar terhadap Weekly Sales berdasarkan korelasi.

2. Semakin besar nilai absolut korelasi, semakin besar hubungan antara variabel tersebut dengan Weekly Sales.

3. Variabel dengan korelasi kecil bukan berarti tidak digunakan oleh model.

4. Random Forest mampu mempelajari hubungan yang bersifat **non-linear**, sehingga variabel yang memiliki korelasi rendah masih dapat memberikan kontribusi terhadap hasil prediksi.

5. Oleh karena itu, seluruh variabel seperti:

- Store
- Holiday Flag
- Temperature
- Fuel Price
- CPI
- Unemployment
- Year
- Month
- Week

tetap digunakan pada proses pelatihan model Random Forest Regression.
""")

    st.warning("""
Catatan:

Korelasi hanya menunjukkan hubungan linier antar variabel.

Sedangkan Random Forest Regression mampu mempelajari hubungan yang
lebih kompleks (non-linear), sehingga hasil Feature Importance dapat
berbeda dengan nilai korelasi.
""")

    st.divider()

    # ======================================================
    # EVALUASI MODEL
    # ======================================================

    st.header("Evaluasi Model Random Forest")

    st.markdown("""
Evaluasi model dilakukan untuk mengetahui seberapa baik model
Random Forest Regression dalam memprediksi nilai **Weekly Sales**.

Semakin kecil nilai **MAE** dan **RMSE**, serta semakin mendekati **1**
nilai **R²**, maka semakin baik performa model.
""")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "MAE",
        f"USD ${mae:,.2f}"
    )

    col2.metric(
        "RMSE",
        f"USD ${rmse:,.2f}"
    )

    col3.metric(
        "R² Score",
        f"{r2:.4f}"
    )

    st.info(f"""
### Hasil Evaluasi

✅ MAE  : **USD ${mae:,.2f}**

✅ RMSE : **USD ${rmse:,.2f}**

✅ R²   : **{r2:.4f}**
""")

    st.divider()

    # ======================================================
    # DATA AKTUAL VS PREDIKSI
    # ======================================================

    st.subheader("Perbandingan Nilai Aktual dan Prediksi")

    hasil_prediksi = pd.DataFrame({
        "Actual Weekly Sales": y_test.values,
        "Prediction Weekly Sales": pred
    })

    st.dataframe(
        style_table(hasil_prediksi.head(20)),
        use_container_width=True
    )

    st.caption(
        "Menampilkan 20 data pertama hasil prediksi model."
    )

    st.divider()

    # ======================================================
    # SCATTER PLOT
    # ======================================================

    st.subheader("Scatter Plot: Actual vs Prediction")

    fig5, ax5 = plt.subplots(figsize=(7,7))

    ax5.scatter(
        y_test,
        pred,
        alpha=0.6
    )

    min_val = min(
        y_test.min(),
        pred.min()
    )

    max_val = max(
        y_test.max(),
        pred.max()
    )

    ax5.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--"
    )

    ax5.set_xlabel("Actual Weekly Sales (USD)")
    ax5.set_ylabel("Prediction Weekly Sales (USD)")
    ax5.set_title("Actual vs Prediction")

    st.pyplot(fig5)

    st.info("""
### Insight Scatter Plot

- Setiap titik menunjukkan satu data uji.
- Semakin dekat titik terhadap garis diagonal,
  maka hasil prediksi semakin mendekati nilai aktual.
- Penyebaran titik yang rapat di sekitar garis
  menunjukkan model memiliki performa yang baik.
""")

    st.divider()

    # ======================================================
    # RESIDUAL ERROR
    # ======================================================

    residual = y_test - pred

    st.subheader("Residual Error")

    residual_df = pd.DataFrame({
        "Actual": y_test.values,
        "Prediction": pred,
        "Residual Error": residual
    })

    st.dataframe(
        style_table(residual_df.head(20)),
        use_container_width=True
    )

    st.caption(
        "Residual Error = Actual - Prediction"
    )

    st.divider()

    # ======================================================
    # HISTOGRAM RESIDUAL
    # ======================================================

    st.subheader("Distribusi Residual Error")

    fig6, ax6 = plt.subplots(figsize=(8,4))

    ax6.hist(
        residual,
        bins=25,
        edgecolor="black"
    )

    ax6.set_xlabel("Residual Error")
    ax6.set_ylabel("Frekuensi")
    ax6.set_title("Histogram Residual Error")

    st.pyplot(fig6)

    st.info("""
### Insight Residual Error

- Residual Error merupakan selisih antara nilai aktual dan hasil prediksi.
- Nilai residual yang mendekati nol menunjukkan prediksi yang baik.
- Distribusi residual yang menyebar di sekitar nol mengindikasikan model tidak memiliki bias yang besar.
""")

    st.divider()

    # ======================================================
    # KESIMPULAN EVALUASI
    # ======================================================

    st.subheader("Kesimpulan Evaluasi Model")

    st.success(f"""
1. Model menggunakan algoritma **Random Forest Regression**.

2. Nilai **MAE** sebesar **USD ${mae:,.2f}** menunjukkan rata-rata kesalahan prediksi model.

3. Nilai **RMSE** sebesar **USD ${rmse:,.2f}** menunjukkan besarnya kesalahan prediksi dengan memberikan penalti lebih besar terhadap error yang tinggi.

4. Nilai **R²** sebesar **{r2:.4f}** menunjukkan kemampuan model dalam menjelaskan variasi data Weekly Sales.

5. Scatter Plot memperlihatkan bahwa sebagian besar hasil prediksi berada cukup dekat dengan nilai aktual.

6. Distribusi Residual Error menunjukkan bahwa kesalahan prediksi tidak terpusat pada nilai tertentu, sehingga model memiliki performa yang cukup baik untuk digunakan dalam melakukan prediksi Weekly Sales.
""")

    st.divider()

    # ======================================================
    # FEATURE IMPORTANCE
    # ======================================================

    st.header("Feature Importance")

    st.markdown("""
Feature Importance menunjukkan tingkat kontribusi masing-masing
variabel terhadap hasil prediksi **Weekly Sales**.

Semakin besar nilai importance, semakin besar pengaruh variabel
tersebut dalam proses pengambilan keputusan oleh Random Forest.
""")

    # ======================================================
    # GRAFIK FEATURE IMPORTANCE
    # ======================================================

    fig7, ax7 = plt.subplots(figsize=(10,6))

    ax7.barh(
        feature_importance["Feature"],
        feature_importance["Importance"]
    )

    ax7.invert_yaxis()

    ax7.set_xlabel("Importance Score")
    ax7.set_ylabel("Feature")
    ax7.set_title("Feature Importance Random Forest")

    st.pyplot(fig7)

    st.divider()

    # ======================================================
    # TABEL FEATURE IMPORTANCE
    # ======================================================

    importance_df = feature_importance.copy()

    importance_df["Persentase (%)"] = (
        importance_df["Importance"] * 100
    )

    importance_df["Ranking"] = range(
        1,
        len(importance_df) + 1
    )

    importance_df = importance_df[
        [
            "Ranking",
            "Feature",
            "Importance",
            "Persentase (%)"
        ]
    ]

    st.subheader("Tabel Feature Importance")

    st.dataframe(
        importance_df.style.format({
            "Importance":"{:.4f}",
            "Persentase (%)":"{:.2f}%"
        }),
        use_container_width=True
    )

    st.divider()

    # ======================================================
    # FEATURE PALING BERPENGARUH
    # ======================================================

    fitur_tertinggi = importance_df.iloc[0]
    fitur_terendah = importance_df.iloc[-1]

    col1, col2 = st.columns(2)

    col1.metric(
        "Feature Paling Berpengaruh",
        fitur_tertinggi["Feature"]
    )

    col2.metric(
        "Importance Score",
        f"{fitur_tertinggi['Importance']:.4f}"
    )

    st.success(f"""
Variabel **{fitur_tertinggi['Feature']}** merupakan fitur yang
paling berpengaruh terhadap prediksi Weekly Sales dengan nilai
importance sebesar **{fitur_tertinggi['Importance']:.4f}**
atau sekitar **{fitur_tertinggi['Persentase (%)']:.2f}%**.
""")

    st.divider()

    # ======================================================
    # FEATURE PALING KECIL
    # ======================================================

    st.warning(f"""
Variabel **{fitur_terendah['Feature']}**
memiliki nilai importance paling kecil yaitu
**{fitur_terendah['Importance']:.4f}**.

Hal ini menunjukkan bahwa kontribusinya terhadap prediksi
lebih kecil dibandingkan variabel lainnya.
""")

    st.divider()

    # ======================================================
    # INTERPRETASI FEATURE IMPORTANCE
    # ======================================================

    st.subheader("Interpretasi Feature Importance")

    interpretasi = pd.DataFrame({

        "Kategori":[
            "Tinggi",
            "Sedang",
            "Rendah"
        ],

        "Keterangan":[
            "Variabel sangat berpengaruh terhadap prediksi.",
            "Variabel cukup membantu proses prediksi.",
            "Variabel tetap digunakan tetapi pengaruhnya relatif kecil."
        ]

    })

    st.dataframe(
        style_table(interpretasi),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ======================================================
    # KESIMPULAN FEATURE IMPORTANCE
    # ======================================================

    st.subheader("Kesimpulan Feature Importance")

    st.success(f"""
1. Random Forest menghitung tingkat kepentingan setiap variabel secara otomatis.

2. Variabel yang paling berpengaruh adalah **{fitur_tertinggi['Feature']}**
   dengan nilai importance sebesar **{fitur_tertinggi['Importance']:.4f}**.

3. Variabel dengan nilai importance kecil tidak berarti tidak berguna,
   tetapi kontribusinya lebih rendah dibandingkan fitur lainnya.

4. Seluruh variabel tetap digunakan oleh model karena Random Forest
   memanfaatkan kombinasi seluruh fitur untuk menghasilkan prediksi yang optimal.

5. Hasil Feature Importance dapat berbeda dengan hasil analisis korelasi,
   karena Random Forest mampu mempelajari hubungan yang bersifat
   **non-linear**, sedangkan korelasi hanya mengukur hubungan linier.

6. Berdasarkan hasil evaluasi model, korelasi, dan Feature Importance,
   Random Forest Regression dinilai layak digunakan untuk memprediksi
   **Weekly Sales (USD)** pada dataset Walmart.
""")

# ==========================================================
# MENU PREDIKSI
# ==========================================================

elif menu == "Prediksi":

    st.header("📈 Prediksi Weekly Sales")

    st.markdown("""
Menu ini digunakan untuk memperkirakan **Weekly Sales (USD)** berdasarkan
nilai input yang diberikan pengguna.

Model yang digunakan adalah **Random Forest Regression** yang telah
dilatih menggunakan data Walmart periode **2011–2012**.
""")

    st.info("""
### Tutorial Penggunaan

1. Isi seluruh data sesuai kondisi yang akan diprediksi.
2. Pilih Store.
3. Pilih Holiday Flag.
4. Isi Temperature, Fuel Price, CPI, dan Unemployment.
5. Masukkan Tahun (boleh lebih dari 2012).
6. Pilih Month.
7. Pilih Week sesuai Month.
8. Klik tombol **Prediksi Weekly Sales**.
""")

    st.warning("""
**Catatan**

- Dataset pelatihan hanya mencakup tahun **2011–2012**.
- Tahun di atas 2012 tetap diperbolehkan sebagai input.
- Prediksi yang dihasilkan merupakan estimasi berdasarkan pola data
  tahun 2011–2012, sehingga bukan merupakan nilai aktual.
""")

    st.divider()

    # ======================================================
    # CONTOH INPUT
    # ======================================================

    with st.expander("Lihat Contoh Pengisian"):

        contoh = pd.DataFrame({

            "Variabel":[
                "Store",
                "Holiday Flag",
                "Temperature",
                "Fuel Price",
                "CPI",
                "Unemployment",
                "Year",
                "Month",
                "Week"
            ],

            "Contoh":[
                5,
                0,
                75.5,
                3.65,
                212.45,
                6.20,
                2026,
                8,
                2
            ]

        })

        st.dataframe(
            style_table(contoh),
            hide_index=True,
            use_container_width=True
        )

    st.divider()

    # ======================================================
    # INPUT
    # ======================================================

    col1, col2, col3 = st.columns(3)

    store = col1.number_input(
        "Store",
        min_value=int(df["Store"].min()),
        max_value=int(df["Store"].max()),
        value=5
    )

    holiday = col2.selectbox(
        "Holiday Flag",
        [0,1],
        format_func=lambda x:
        "Hari Libur" if x==1 else "Bukan Hari Libur"
    )

    temperature = col3.number_input(
        "Temperature",
        min_value=float(df["Temperature"].min()),
        max_value=float(df["Temperature"].max()),
        value=float(df["Temperature"].mean())
    )

    fuel = col1.number_input(
        "Fuel Price",
        min_value=float(df["Fuel_Price"].min()),
        max_value=float(df["Fuel_Price"].max()),
        value=float(df["Fuel_Price"].mean())
    )

    cpi = col2.number_input(
        "CPI",
        min_value=float(df["CPI"].min()),
        max_value=float(df["CPI"].max()),
        value=float(df["CPI"].mean())
    )

    unemployment = col3.number_input(
        "Unemployment",
        min_value=float(df["Unemployment"].min()),
        max_value=float(df["Unemployment"].max()),
        value=float(df["Unemployment"].mean())
    )

    year = col1.number_input(
        "Year",
        min_value=2011,
        max_value=2100,
        value=2012
    )

    month = col2.selectbox(
        "Month",
        list(range(1,13))
    )

    minggu_bulan = {
        1:5,
        2:4,
        3:5,
        4:4,
        5:5,
        6:4,
        7:5,
        8:5,
        9:4,
        10:5,
        11:4,
        12:5
    }

    max_week = minggu_bulan[month]

    week = col3.selectbox(
        "Week",
        list(range(1,max_week+1))
    )

    st.caption(
        f"Bulan {month} hanya memiliki pilihan minggu 1–{max_week}."
    )

    st.divider()

    # ======================================================
    # PREDIKSI
    # ======================================================

    if st.button("Prediksi Weekly Sales"):

        data_baru = pd.DataFrame({

            "Store":[store],
            "Holiday_Flag":[holiday],
            "Temperature":[temperature],
            "Fuel_Price":[fuel],
            "CPI":[cpi],
            "Unemployment":[unemployment],
            "Year":[year],
            "Month":[month],
            "Week":[week]

        })

        hasil = model.predict(data_baru)[0]

        # ==============================================
        # KATEGORI
        # ==============================================

        q1 = df["Weekly_Sales"].quantile(0.25)
        q3 = df["Weekly_Sales"].quantile(0.75)

        if hasil < q1:
            kategori = "🔴 Rendah"
        elif hasil < q3:
            kategori = "🟡 Sedang"
        else:
            kategori = "🟢 Tinggi"

        st.success(
            f"Estimasi Weekly Sales : **USD ${hasil:,.2f}**"
        )

        col1,col2 = st.columns(2)

        col1.metric(
            "Prediksi Weekly Sales",
            f"USD ${hasil:,.2f}"
        )

        col2.metric(
            "Kategori Penjualan",
            kategori
        )

        st.divider()

        st.subheader("Penjelasan Hasil")

        st.info(f"""
Model memperkirakan bahwa nilai **Weekly Sales**
adalah sekitar **USD ${hasil:,.2f}**.

Kategori hasil prediksi adalah **{kategori}**.

Prediksi ini diperoleh menggunakan algoritma
**Random Forest Regression** berdasarkan pola
data historis Walmart periode **2011–2012**.
""")

        st.divider()

        st.subheader("Cara Kerja Random Forest")

        st.markdown("""
1. Model menerima seluruh nilai input.
2. Data diproses oleh banyak Decision Tree.
3. Setiap Decision Tree menghasilkan prediksi.
4. Semua hasil prediksi dirata-ratakan.
5. Nilai rata-rata menjadi hasil prediksi Weekly Sales.

Karena menggunakan banyak pohon keputusan (Decision Tree),
Random Forest cenderung menghasilkan prediksi yang lebih stabil
dibandingkan hanya menggunakan satu Decision Tree.
""")

# ==========================================================
# MENU KESIMPULAN
# ==========================================================

elif menu == "Kesimpulan":

    st.header("📄 Kesimpulan")

    st.markdown("""
Menu ini berisi ringkasan seluruh proses mulai dari analisis dataset,
analisis statistik, pembangunan model Machine Learning hingga proses
prediksi Weekly Sales.
""")

    st.divider()

    # ======================================================
    # RINGKASAN DATASET
    # ======================================================

    st.subheader("1. Ringkasan Dataset")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Jumlah Data",
        f"{len(df):,}"
    )

    col2.metric(
        "Jumlah Store",
        df["Store"].nunique()
    )

    col3.metric(
        "Periode Data",
        "2011 - 2012"
    )

    st.success("""
Dataset Walmart berhasil dimuat dan diproses dengan baik.

Dataset terdiri dari data penjualan mingguan (Weekly Sales)
yang berasal dari beberapa Store Walmart selama periode
2011–2012.
""")

    st.divider()

    # ======================================================
    # RINGKASAN STATISTIK
    # ======================================================

    st.subheader("2. Ringkasan Statistik")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Rata-rata",
        f"USD ${df['Weekly_Sales'].mean():,.2f}"
    )

    col2.metric(
        "Minimum",
        f"USD ${df['Weekly_Sales'].min():,.2f}"
    )

    col3.metric(
        "Maksimum",
        f"USD ${df['Weekly_Sales'].max():,.2f}"
    )

    st.info("""
Analisis statistik menunjukkan bahwa Weekly Sales memiliki
variasi yang cukup besar antar Store, sehingga pendekatan
Machine Learning diperlukan untuk mempelajari pola penjualan.
""")

    st.divider()

    # ======================================================
    # RINGKASAN MACHINE LEARNING
    # ======================================================

    st.subheader("3. Ringkasan Machine Learning")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "MAE",
        f"USD ${mae:,.2f}"
    )

    col2.metric(
        "RMSE",
        f"USD ${rmse:,.2f}"
    )

    col3.metric(
        "R² Score",
        f"{r2:.4f}"
    )

    st.success(f"""
Model menggunakan algoritma **Random Forest Regression**.

Split data terbaik adalah **{best_split['Split']}**
berdasarkan nilai R².

Model mampu memprediksi Weekly Sales dengan performa yang
ditunjukkan oleh nilai MAE, RMSE, dan R² di atas.
""")

    st.divider()

    # ======================================================
    # FEATURE TERPENTING
    # ======================================================

    st.subheader("4. Variabel Paling Berpengaruh")

    fitur = feature_importance.iloc[0]

    st.metric(
        "Feature Terpenting",
        fitur["Feature"]
    )

    st.metric(
        "Importance Score",
        f"{fitur['Importance']:.4f}"
    )

    st.info(f"""
Variabel **{fitur['Feature']}**
merupakan variabel yang paling berpengaruh terhadap
hasil prediksi Weekly Sales berdasarkan Feature Importance
yang dihitung oleh Random Forest Regression.
""")

    st.divider()

    # ======================================================
    # KESIMPULAN AKHIR
    # ======================================================

    st.subheader("5. Kesimpulan Akhir")

    st.success(f"""
1. Dataset Walmart periode **2011–2012** berhasil digunakan
   sebagai dasar pembangunan model prediksi Weekly Sales.

2. Hasil analisis statistik menunjukkan adanya variasi
   penjualan antar Store sehingga pendekatan Machine Learning
   layak digunakan.

3. Algoritma **Random Forest Regression** dipilih karena
   mampu menangani hubungan non-linear, stabil terhadap
   data, serta dapat menghitung Feature Importance.

4. Berdasarkan hasil evaluasi, model memperoleh:

   • MAE  : USD ${mae:,.2f}

   • RMSE : USD ${rmse:,.2f}

   • R²   : {r2:.4f}

5. Variabel yang paling berpengaruh terhadap prediksi adalah
   **{fitur['Feature']}**.

6. Dashboard ini memungkinkan pengguna melakukan simulasi
   prediksi Weekly Sales dengan memasukkan kondisi yang
   diinginkan.

7. Walaupun pengguna dapat memasukkan tahun di atas 2012,
   model tetap melakukan prediksi berdasarkan pola data
   historis periode **2011–2012**, sehingga hasil prediksi
   merupakan estimasi dan bukan nilai aktual.

8. Dashboard ini diharapkan dapat membantu dalam proses
   analisis data serta mendukung pengambilan keputusan
   berdasarkan hasil prediksi penjualan.
""")

    st.balloons()
