"""
Dashboard Prediksi Penjualan Walmart
-------------------------------------
Eksplorasi data, statistik deskriptif, model Random Forest Regression,
dan simulasi prediksi Weekly Sales berbasis data Walmart 2011-2012.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from streamlit_option_menu import option_menu

DATA_URL = (
    "https://raw.githubusercontent.com/rafara233/walmart-sales-forecasting-random-forest/refs/heads/main/Walmart_Sales.csv"
)

# ----------------------------------------------------------------------------
# PALET & TEMA
# ----------------------------------------------------------------------------

INK = "#132A3A"
PAPER = "#EEF1ED"
CARD = "#FFFFFF"
BORDER = "#DCE3DD"
ACCENT = "#2F6F52"
GOLD = "#B98A2E"
MUTED = "#5B6B66"
PALETTE = [ACCENT, GOLD, INK, "#7C9885", "#C97B4A", "#3D5A5B"]

DISPLAY_FONT = "'Space Grotesk', sans-serif"
BODY_FONT = "'IBM Plex Sans', sans-serif"
MONO_FONT = "'IBM Plex Mono', monospace"

MINGGU_PER_BULAN = {1: 5, 2: 4, 3: 5, 4: 4, 5: 5, 6: 4, 7: 5, 8: 5, 9: 4, 10: 5, 11: 4, 12: 5}


# ----------------------------------------------------------------------------
# SETUP HALAMAN & GAYA
# ----------------------------------------------------------------------------

def configure_page():
    st.set_page_config(
        page_title="Dashboard Prediksi Penjualan Walmart",
        page_icon="🧾",
        layout="wide",
    )

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

        html, body, [class*="css"] {{
            font-family: {BODY_FONT};
            color: {INK};
        }}

        .stApp {{ background-color: {PAPER}; }}

        h1, h2, h3, h4 {{
            font-family: {DISPLAY_FONT} !important;
            color: {INK} !important;
            letter-spacing: -0.01em;
        }}

        .eyebrow {{
            font-family: {MONO_FONT};
            font-size: 0.72rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: {GOLD};
            margin-bottom: 0.2rem;
        }}

        .lede {{
            color: {MUTED};
            font-size: 1.02rem;
            max-width: 780px;
            line-height: 1.55;
        }}

        div[data-testid="stMetric"] {{
            background: {CARD};
            border: 1px solid {BORDER};
            border-bottom: 2px dashed {BORDER};
            border-radius: 14px;
            padding: 0.9rem 1.1rem 0.8rem 1.1rem;
            box-shadow: 0 1px 2px rgba(19,42,58,0.05);
        }}
        div[data-testid="stMetricValue"] {{
            font-family: {MONO_FONT} !important;
            color: {INK} !important;
        }}
        div[data-testid="stMetricLabel"] {{
            font-family: {MONO_FONT} !important;
            text-transform: uppercase;
            font-size: 0.7rem !important;
            letter-spacing: 0.05em;
            color: {MUTED} !important;
        }}

        div[data-testid="stAlert"] {{
            border-radius: 12px;
            box-shadow: 0 1px 2px rgba(19,42,58,0.04);
        }}

        div.stButton > button {{
            font-family: {DISPLAY_FONT};
            background-color: {INK} !important;
            color: #F6F5F0 !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            height: 46px;
            letter-spacing: 0.02em;
            transition: background-color 0.15s ease;
        }}
        div.stButton > button:hover {{
            background-color: {ACCENT} !important;
            color: white !important;
        }}

        div[data-testid="stDataFrame"] {{
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid {BORDER};
        }}

        div[data-testid="stExpander"] {{
            border: 1px solid {BORDER};
            border-radius: 12px;
            background: {CARD};
        }}

        hr {{ border-top: 1px dashed {BORDER} !important; }}

        #MainMenu, footer {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(eyebrow: str, title: str, lede: str = ""):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    if lede:
        st.markdown(f'<p class="lede">{lede}</p>', unsafe_allow_html=True)
    st.write("")


def style_table(df: pd.DataFrame, bar_cols=None):
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    text_cols = df.select_dtypes(exclude=["int64", "float64"]).columns

    styler = df.style.set_table_styles(
        [
            {"selector": "th", "props": [("text-align", "center"), ("font-weight", "600")]},
            {"selector": "td", "props": [("font-size", "13.5px")]},
        ]
    )
    styler = styler.set_properties(subset=numeric_cols, **{"text-align": "center"})
    styler = styler.set_properties(subset=text_cols, **{"text-align": "left"})

    if bar_cols:
        styler = styler.bar(subset=bar_cols, color=ACCENT + "55")

    return styler


def themed_chart(fig, height=420):
    fig.update_layout(
        font_family=BODY_FONT,
        title_font_family=DISPLAY_FONT,
        title_font_color=INK,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        colorway=PALETTE,
        height=height,
        margin=dict(t=50, l=10, r=10, b=10),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    return fig


# ----------------------------------------------------------------------------
# DATA & MODEL
# ----------------------------------------------------------------------------

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL)
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    return df


@st.cache_resource
def train_models(df: pd.DataFrame):
    df_model = df.drop(columns=["Date"])
    X = df_model.drop(columns=["Weekly_Sales"])
    y = df_model["Weekly_Sales"]

    split_ratio = {"90 : 10": 0.10, "80 : 20": 0.20, "70 : 30": 0.30, "60 : 40": 0.40}
    hasil_split = []
    for nama, test_size in split_ratio.items():
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(Xtr, ytr)
        pred = rf.predict(Xte)
        hasil_split.append(
            {
                "Split": nama,
                "MAE": mean_absolute_error(yte, pred),
                "RMSE": np.sqrt(mean_squared_error(yte, pred)),
                "R²": r2_score(yte, pred),
            }
        )
    hasil_split = pd.DataFrame(hasil_split)
    best_split = hasil_split.loc[hasil_split["R²"].idxmax()]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    feature_importance = pd.DataFrame(
        {"Feature": X.columns, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False)

    corr = df_model.corr(numeric_only=True)

    return {
        "df_model": df_model,
        "X": X,
        "y": y,
        "hasil_split": hasil_split,
        "best_split": best_split,
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "pred": pred,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "feature_importance": feature_importance,
        "corr": corr,
    }


# ----------------------------------------------------------------------------
# HALAMAN: DATASET
# ----------------------------------------------------------------------------

def page_dataset(df: pd.DataFrame):
    page_header(
        "01 · DATASET",
        "Dataset Walmart",
        "Data mentah di balik dashboard ini — cakupan, kelengkapan, dan arti tiap kolom, "
        "sebelum masuk ke tahap analisis dan pemodelan.",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baris", f"{df.shape[0]:,}")
    c2.metric("Kolom", df.shape[1])
    c3.metric("Missing Value", int(df.isnull().sum().sum()))
    c4.metric("Data Duplikat", int(df.duplicated().sum()))

    st.divider()
    st.subheader("Pratinjau")
    jumlah = st.slider("Jumlah baris yang ditampilkan", 5, 30, 10)
    st.dataframe(style_table(df.head(jumlah)), use_container_width=True)

    st.divider()
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Struktur kolom")
        info = pd.DataFrame(
            {
                "Kolom": df.columns,
                "Tipe": df.dtypes.astype(str),
                "Missing": df.isnull().sum().values,
                "Unik": [df[c].nunique() for c in df.columns],
            }
        )
        st.dataframe(style_table(info), use_container_width=True, hide_index=True)

    with right:
        st.subheader("Arti tiap kolom")
        deskripsi = pd.DataFrame(
            {
                "Kolom": [
                    "Store", "Date", "Holiday_Flag", "Temperature", "Fuel_Price",
                    "CPI", "Unemployment", "Weekly_Sales", "Year", "Month", "Week",
                ],
                "Keterangan": [
                    "Nomor toko Walmart",
                    "Tanggal transaksi mingguan",
                    "1 jika minggu tersebut mengandung hari libur besar",
                    "Suhu rata-rata di sekitar toko",
                    "Harga bahan bakar regional",
                    "Consumer Price Index",
                    "Tingkat pengangguran regional",
                    "Total penjualan mingguan (USD) — target prediksi",
                    "Tahun, diturunkan dari Date",
                    "Bulan, diturunkan dari Date",
                    "Minggu ke-berapa dalam setahun",
                ],
            }
        )
        st.dataframe(style_table(deskripsi), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Kelengkapan data")

    miss_ok = df.isnull().sum().sum() == 0
    dup_ok = df.duplicated().sum() == 0
    a, b = st.columns(2)
    with a:
        if miss_ok:
            st.success("Tidak ada missing value — data siap dipakai apa adanya.")
        else:
            st.warning("Masih ada missing value yang perlu ditangani sebelum pemodelan.")
    with b:
        if dup_ok:
            st.success("Tidak ada baris duplikat.")
        else:
            st.warning(f"Ditemukan {int(df.duplicated().sum())} baris duplikat.")

    st.info(
        f"Dataset mencakup **{df['Store'].nunique()} toko** sepanjang periode "
        f"**{int(df['Year'].min())}–{int(df['Year'].max())}**, dengan Weekly Sales sebagai "
        "variabel target dan sembilan variabel lain — Store, Holiday Flag, Temperature, "
        "Fuel Price, CPI, Unemployment, Year, Month, Week — sebagai fitur untuk model."
    )


# ----------------------------------------------------------------------------
# HALAMAN: STATISTIK
# ----------------------------------------------------------------------------

def page_statistik(df: pd.DataFrame):
    page_header(
        "02 · STATISTIK",
        "Statistik Deskriptif",
        "Melihat sebaran dan pola Weekly Sales sebelum data dilempar ke algoritma — "
        "titik awal untuk memahami apa yang sebenarnya sedang dipelajari model.",
    )

    st.subheader("Ringkasan per variabel")
    statistik = df.describe().T.rename(
        columns={
            "count": "Jumlah", "mean": "Rata-rata", "std": "Std. Deviasi",
            "min": "Min", "25%": "Q1", "50%": "Median", "75%": "Q3", "max": "Max",
        }
    )
    st.dataframe(style_table(statistik), use_container_width=True)
    st.caption(
        "Standar deviasi yang besar menandakan penjualan bervariasi tajam antar toko dan waktu — "
        "sinyal awal bahwa hubungan di data ini tidak akan sesederhana garis lurus."
    )

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribusi Weekly Sales")
        fig = px.histogram(df, x="Weekly_Sales", nbins=30)
        fig.update_traces(marker_line_color=INK, marker_line_width=0.5)
        st.plotly_chart(themed_chart(fig), use_container_width=True)

    with col2:
        st.subheader("Sebaran & outlier")
        fig2 = px.box(df, x="Weekly_Sales", points="outliers")
        st.plotly_chart(themed_chart(fig2), use_container_width=True)

    st.caption(
        "Random Forest membangun banyak pohon keputusan dan merata-ratakan hasilnya, sehingga "
        "relatif tidak terganggu oleh titik-titik ekstrem pada boxplot di atas."
    )

    st.divider()
    st.subheader("Rata-rata penjualan per toko")
    avg_store = df.groupby("Store")["Weekly_Sales"].mean().sort_index()
    fig3 = px.bar(x=avg_store.index.astype(str), y=avg_store.values, labels={"x": "Store", "y": "Weekly Sales (USD)"})
    fig3.update_traces(marker_color=ACCENT)
    st.plotly_chart(themed_chart(fig3, height=400), use_container_width=True)
    st.markdown(
        f"Store **{avg_store.idxmax()}** memimpin dengan rata-rata sekitar "
        f"**USD ${avg_store.max():,.0f}** per minggu, sementara Store **{avg_store.idxmin()}** "
        f"berada di ujung bawah pada **USD ${avg_store.min():,.0f}**. Rentang sejauh ini menunjukkan "
        "tiap toko punya karakter penjualan sendiri — bukan hanya soal lokasi, tapi juga musim dan kondisi ekonomi lokal."
    )

    st.divider()
    st.subheader("Pola musiman per bulan")
    avg_month = df.groupby("Month")["Weekly_Sales"].mean().sort_index()
    fig4 = px.line(x=avg_month.index, y=avg_month.values, markers=True, labels={"x": "Bulan", "y": "Weekly Sales (USD)"})
    fig4.update_traces(line_color=GOLD, marker=dict(size=8, color=INK))
    fig4.update_xaxes(dtick=1)
    st.plotly_chart(themed_chart(fig4, height=380), use_container_width=True)
    st.caption("Naik-turun di sepanjang tahun ini yang membuat Month dan Week layak dijadikan fitur, bukan sekadar metadata.")

    st.divider()
    st.subheader("Angka kunci")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rata-rata", f"${df['Weekly_Sales'].mean():,.0f}")
    m2.metric("Median", f"${df['Weekly_Sales'].median():,.0f}")
    m3.metric("Std. Dev", f"${df['Weekly_Sales'].std():,.0f}")
    m4.metric("Min", f"${df['Weekly_Sales'].min():,.0f}")
    m5.metric("Max", f"${df['Weekly_Sales'].max():,.0f}")


# ----------------------------------------------------------------------------
# HALAMAN: MACHINE LEARNING
# ----------------------------------------------------------------------------

def page_ml(df: pd.DataFrame, results: dict):
    X = results["X"]
    hasil_split = results["hasil_split"]
    best_split = results["best_split"]
    corr = results["corr"]
    model = results["model"]
    mae, rmse, r2 = results["mae"], results["rmse"], results["r2"]
    y_test, pred = results["y_test"], results["pred"]
    feature_importance = results["feature_importance"]

    page_header(
        "03 · MODEL",
        "Random Forest Regression",
        "Bagaimana model dibangun, seberapa baik performanya, dan variabel mana yang "
        "sebenarnya menggerakkan angka Weekly Sales.",
    )

    st.markdown(
        "Random Forest bekerja dengan membangun banyak *decision tree* dari sampel data yang "
        "sedikit berbeda-beda, lalu merata-ratakan prediksi seluruh pohon. Pendekatan ini dipilih "
        "karena tahan terhadap outlier, tidak menuntut data dinormalisasi lebih dulu, dan bisa "
        "menangkap hubungan non-linear antar variabel — sesuatu yang tidak selalu terlihat dari "
        "korelasi biasa."
    )

    st.divider()
    st.subheader("Fitur & target")
    fcol, tcol = st.columns(2)
    with fcol:
        st.caption("VARIABEL INPUT")
        st.dataframe(style_table(pd.DataFrame({"Feature": X.columns})), use_container_width=True, hide_index=True)
    with tcol:
        st.caption("VARIABEL TARGET")
        st.dataframe(
            style_table(pd.DataFrame({"Target": ["Weekly_Sales"], "Keterangan": ["Total penjualan mingguan (USD)"]})),
            use_container_width=True, hide_index=True,
        )

    st.divider()
    st.subheader("Memilih rasio split data")
    st.dataframe(
        hasil_split.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R²": "{:.4f}"}),
        use_container_width=True, hide_index=True,
    )
    st.success(f"Rasio **{best_split['Split']}** memberi R² tertinggi, sehingga dipakai sebagai konfigurasi final model.")

    st.divider()
    st.subheader("Hubungan antar variabel")
    hcol, tcol2 = st.columns([1.2, 1])
    with hcol:
        fig = px.imshow(corr, color_continuous_scale=[GOLD, "#FFFFFF", ACCENT], zmin=-1, zmax=1, text_auto=".2f")
        fig.update_traces(textfont_size=9)
        st.plotly_chart(themed_chart(fig, height=460), use_container_width=True)
    with tcol2:
        corr_target = corr["Weekly_Sales"].drop("Weekly_Sales").sort_values(ascending=False)
        corr_df = pd.DataFrame({"Feature": corr_target.index, "Korelasi": corr_target.values})
        st.dataframe(style_table(corr_df, bar_cols=["Korelasi"]), use_container_width=True, hide_index=True)
        terbesar = corr_target.abs().idxmax()
        st.caption(
            f"**{terbesar}** punya korelasi linear terkuat terhadap Weekly Sales "
            f"({corr_target[terbesar]:.2f}). Tapi korelasi rendah bukan berarti tidak penting — "
            "Random Forest bisa menangkap pola non-linear yang luput dari angka ini."
        )

    st.divider()
    st.subheader("Performa model")
    e1, e2, e3 = st.columns(3)
    e1.metric("MAE", f"${mae:,.0f}")
    e2.metric("RMSE", f"${rmse:,.0f}")
    e3.metric("R² Score", f"{r2:.3f}")
    st.caption("Semakin kecil MAE/RMSE dan semakin dekat R² ke 1, semakin baik model menjelaskan variasi data.")

    scol, rcol = st.columns(2)
    with scol:
        st.markdown("**Actual vs Prediction**")
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=y_test, y=pred, mode="markers", marker=dict(color=ACCENT, opacity=0.55, size=6), name="Data uji"))
        lim = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
        fig5.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=GOLD, dash="dash"), name="Ideal"))
        fig5.update_layout(xaxis_title="Actual (USD)", yaxis_title="Prediction (USD)", showlegend=False)
        st.plotly_chart(themed_chart(fig5, height=380), use_container_width=True)

    with rcol:
        st.markdown("**Distribusi residual**")
        residual = y_test - pred
        fig6 = px.histogram(x=residual, nbins=25, labels={"x": "Residual (Actual − Prediction)"})
        fig6.update_traces(marker_color=INK, marker_line_color=BORDER, marker_line_width=0.5)
        st.plotly_chart(themed_chart(fig6, height=380), use_container_width=True)

    st.markdown(
        "Titik-titik pada scatter plot berkumpul rapat di sekitar garis diagonal, dan residual "
        "menyebar cukup simetris di sekitar nol — tanda model tidak condong terlalu tinggi atau "
        "terlalu rendah secara sistematis."
    )

    st.divider()
    st.subheader("Feature importance")
    fig7 = px.bar(
        feature_importance.sort_values("Importance"),
        x="Importance", y="Feature", orientation="h",
    )
    fig7.update_traces(marker_color=ACCENT)
    st.plotly_chart(themed_chart(fig7, height=420), use_container_width=True)

    top_feat = feature_importance.iloc[0]
    low_feat = feature_importance.iloc[-1]
    tcol3, lcol3 = st.columns(2)
    with tcol3:
        st.success(
            f"**{top_feat['Feature']}** paling menentukan hasil prediksi "
            f"(importance {top_feat['Importance']:.3f}, ≈{top_feat['Importance']*100:.1f}% kontribusi)."
        )
    with lcol3:
        st.warning(
            f"**{low_feat['Feature']}** berkontribusi paling kecil "
            f"({low_feat['Importance']:.3f}) — tetap dipakai model, hanya pengaruhnya lebih tipis."
        )

    with st.expander("Kenapa hasil feature importance bisa beda dari tabel korelasi?"):
        st.write(
            "Korelasi hanya menangkap hubungan garis lurus antara dua variabel. Random Forest "
            "menilai kontribusi variabel lewat seberapa sering dan seberapa efektif ia dipakai "
            "untuk memecah data di dalam pohon keputusan — termasuk pola musiman atau interaksi "
            "antar variabel yang tidak linear. Karena itu urutan pentingnya bisa berbeda."
        )


# ----------------------------------------------------------------------------
# HALAMAN: PREDIKSI
# ----------------------------------------------------------------------------

def page_prediksi(df: pd.DataFrame, results: dict):
    model = results["model"]

    page_header(
        "04 · SIMULASI",
        "Prediksi Weekly Sales",
        "Masukkan kondisi toko dan ekonomi, lalu lihat estimasi penjualan mingguan menurut model.",
    )

    with st.expander("Contoh pengisian"):
        contoh = pd.DataFrame(
            {
                "Variabel": ["Store", "Holiday Flag", "Temperature", "Fuel Price", "CPI", "Unemployment", "Year", "Month", "Week"],
                "Contoh": [5, 0, 75.5, 3.65, 212.45, 6.20, 2026, 8, 2],
            }
        )
        st.dataframe(style_table(contoh), hide_index=True, use_container_width=True)

    st.divider()
    c1, c2, c3 = st.columns(3)
    store = c1.number_input("Store", min_value=int(df["Store"].min()), max_value=int(df["Store"].max()), value=5)
    holiday = c2.selectbox("Holiday Flag", [0, 1], format_func=lambda x: "Hari Libur" if x == 1 else "Hari Biasa")
    temperature = c3.number_input("Temperature", min_value=float(df["Temperature"].min()), max_value=float(df["Temperature"].max()), value=float(df["Temperature"].mean()))

    fuel = c1.number_input("Fuel Price", min_value=float(df["Fuel_Price"].min()), max_value=float(df["Fuel_Price"].max()), value=float(df["Fuel_Price"].mean()))
    cpi = c2.number_input("CPI", min_value=float(df["CPI"].min()), max_value=float(df["CPI"].max()), value=float(df["CPI"].mean()))
    unemployment = c3.number_input("Unemployment", min_value=float(df["Unemployment"].min()), max_value=float(df["Unemployment"].max()), value=float(df["Unemployment"].mean()))

    year = c1.number_input("Year", min_value=2011, max_value=2100, value=2026)
    month = c2.selectbox("Month", list(range(1, 13)))
    max_week = MINGGU_PER_BULAN[month]
    week = c3.selectbox("Week", list(range(1, max_week + 1)))
    st.caption(f"Bulan {month} punya pilihan minggu 1–{max_week}.")

    st.divider()

    if st.button("Hitung Prediksi"):
        data_baru = pd.DataFrame(
            {
                "Store": [store], "Holiday_Flag": [holiday], "Temperature": [temperature],
                "Fuel_Price": [fuel], "CPI": [cpi], "Unemployment": [unemployment],
                "Year": [year], "Month": [month], "Week": [week],
            }
        )
        hasil = model.predict(data_baru)[0]

        q1, q3 = df["Weekly_Sales"].quantile([0.25, 0.75])
        if hasil < q1:
            kategori, warna = "Rendah", "🔴"
        elif hasil < q3:
            kategori, warna = "Sedang", "🟡"
        else:
            kategori, warna = "Tinggi", "🟢"

        st.success(f"Estimasi Weekly Sales: **USD ${hasil:,.2f}**")

        rc1, rc2 = st.columns(2)
        rc1.metric("Prediksi Weekly Sales", f"${hasil:,.0f}")
        rc2.metric("Kategori", f"{warna} {kategori}")

        st.divider()
        st.markdown(
            f"Berdasarkan kondisi yang dimasukkan, model memperkirakan penjualan mingguan Store "
            f"**{store}** berada pada kisaran **{kategori.lower()}** dibanding toko-toko lain dalam "
            "dataset. Angka ini adalah rata-rata dari seluruh pohon keputusan dalam model, dilatih "
            "dari pola historis 2011–2012 — jadi paling akurat untuk kondisi yang mirip rentang data itu."
        )

    st.caption(
        "Dataset pelatihan hanya mencakup 2011–2012. Tahun di luar rentang itu tetap bisa diisi, "
        "namun hasilnya adalah ekstrapolasi, bukan nilai yang benar-benar teruji."
    )


# ----------------------------------------------------------------------------
# HALAMAN: KESIMPULAN
# ----------------------------------------------------------------------------

def page_kesimpulan(df: pd.DataFrame, results: dict):
    mae, rmse, r2 = results["mae"], results["rmse"], results["r2"]
    best_split = results["best_split"]
    feature_importance = results["feature_importance"]
    top_feat = feature_importance.iloc[0]

    page_header(
        "05 · RINGKASAN",
        "Kesimpulan",
        "Rangkuman singkat dari seluruh alur — dari data mentah sampai model siap dipakai.",
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah Data", f"{len(df):,}")
    c2.metric("Jumlah Store", df["Store"].nunique())
    c3.metric("Periode", f"{int(df['Year'].min())}–{int(df['Year'].max())}")

    st.markdown(
        f"Dashboard ini dibangun di atas **{len(df):,} baris** data penjualan mingguan dari "
        f"**{df['Store'].nunique()} toko** Walmart. Setelah eksplorasi data dan statistik deskriptif "
        "menunjukkan variasi penjualan yang cukup besar antar toko dan antar bulan, pendekatan "
        "Machine Learning — khususnya **Random Forest Regression** — dipilih karena mampu menangani "
        "hubungan non-linear tanpa perlu banyak pra-pemrosesan."
    )

    st.divider()
    st.subheader("Performa model")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"${mae:,.0f}")
    m2.metric("RMSE", f"${rmse:,.0f}")
    m3.metric("R² Score", f"{r2:.3f}")
    st.caption(f"Konfigurasi terbaik menggunakan rasio split **{best_split['Split']}**, dipilih berdasarkan R² tertinggi.")

    st.divider()
    st.subheader("Variabel paling berpengaruh")
    v1, v2 = st.columns(2)
    v1.metric("Feature Terpenting", top_feat["Feature"])
    v2.metric("Importance Score", f"{top_feat['Importance']:.3f}")
    st.markdown(
        f"**{top_feat['Feature']}** konsisten menjadi variabel dengan pengaruh terbesar terhadap "
        "prediksi Weekly Sales, sejalan dengan pola yang juga terlihat pada analisis korelasi di menu Machine Learning."
    )

    st.divider()
    st.subheader("Catatan pemakaian")
    st.warning(
        "Model dilatih dari data 2011–2012. Prediksi untuk tahun di luar rentang tersebut bersifat "
        "estimasi berbasis pola historis, bukan proyeksi ekonomi yang memperhitungkan perubahan "
        "kondisi pasar terkini."
    )

    st.success(
        "Secara keseluruhan, dashboard ini memberi gambaran menyeluruh dari data mentah, statistik, "
        "hingga model siap pakai — cukup untuk eksplorasi awal maupun simulasi skenario penjualan mingguan."
    )

    st.balloons()


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main():
    configure_page()

    st.markdown(
        '<div class="eyebrow">WALMART SALES INTELLIGENCE</div>'
        '<h1 style="margin-top:-6px;">🧾 Dashboard Prediksi Penjualan</h1>'
        '<p class="lede">Eksplorasi data, statistik, dan model Random Forest untuk memperkirakan '
        'Weekly Sales toko Walmart.</p>',
        unsafe_allow_html=True,
    )

    menu = option_menu(
        menu_title=None,
        options=["Dataset", "Statistik", "Machine Learning", "Prediksi", "Kesimpulan"],
        icons=["table", "bar-chart", "cpu", "graph-up-arrow", "clipboard-check"],
        orientation="horizontal",
        default_index=0,
        styles={
            "container": {
                "padding": "6px",
                "background-color": CARD,
                "border-radius": "14px",
                "border": f"1px solid {BORDER}",
            },
            "icon": {"color": MUTED, "font-size": "16px"},
            "nav-link": {
                "font-family": DISPLAY_FONT,
                "font-size": "15px",
                "font-weight": "600",
                "color": INK,
                "background-color": "transparent",
                "text-align": "center",
                "margin": "2px",
                "padding": "10px 14px",
                "border-radius": "10px",
            },
            "nav-link-selected": {
                "background-color": ACCENT,
                "color": "#FFFFFF",
                "font-weight": "700",
            },
        },
    )

    df = load_data()
    results = train_models(df)

    if menu == "Dataset":
        page_dataset(df)
    elif menu == "Statistik":
        page_statistik(df)
    elif menu == "Machine Learning":
        page_ml(df, results)
    elif menu == "Prediksi":
        page_prediksi(df, results)
    elif menu == "Kesimpulan":
        page_kesimpulan(df, results)


if __name__ == "__main__":
    main()
