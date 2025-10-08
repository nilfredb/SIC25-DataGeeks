
# app.py — JobPulse Streamlit (fixed 'country_name' and Streamlit plotting)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="JobPulse — Desempleo & PIB", layout="wide")

# =============================
# Carga y preparación de datos
# =============================
@st.cache_data(show_spinner=True)
def load_and_prepare(unemp_path: str, gdp_path: str):
    # Leer datasets
    unemp = pd.read_csv(unemp_path)
    gdp = pd.read_csv(gdp_path)

    # Desempleo: ancho -> largo
    year_cols = unemp.columns[4:]
    unemp_long = unemp.melt(
        id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
        value_vars=year_cols,
        var_name='year',
        value_name='unemployment_rate'
    ).dropna(subset=['unemployment_rate'])

    unemp_long['year'] = pd.to_numeric(unemp_long['year'], errors='coerce').astype('Int64')
    unemp_long.dropna(subset=['year'], inplace=True)
    unemp_long['year'] = unemp_long['year'].astype(int)

    unemp_long.rename(columns={
        'Country Name': 'country_name',
        'Country Code': 'country_code'
    }, inplace=True)

    # PIB por país
    gdp = gdp[['country_code', 'country_name', 'year', 'gdp_per_capita']].copy()
    gdp['country_code'] = gdp['country_code'].astype(str).str.strip().str.upper()
    gdp['country_name'] = gdp['country_name'].astype(str).str.strip()
    gdp['year'] = pd.to_numeric(gdp['year'], errors='coerce').astype('Int64')
    gdp['gdp_per_capita'] = pd.to_numeric(gdp['gdp_per_capita'], errors='coerce')
    gdp.dropna(subset=['year'], inplace=True)
    gdp['year'] = gdp['year'].astype(int)

    # Para evitar columnas duplicadas de nombre de país, renombramos el del PIB
    gdp = gdp.rename(columns={'country_name': 'country_name_gdp'})

    # Merge
    merged = pd.merge(
        unemp_long,
        gdp[['country_code', 'year', 'gdp_per_capita', 'country_name_gdp']],
        on=['country_code', 'year'],
        how='inner'
    )

    # Consolidar 'country_name' (prioriza desempleo; si falta, usa el del PIB)
    if 'country_name' in merged.columns:
        merged['country_name'] = merged['country_name'].fillna(merged['country_name_gdp'])
    else:
        merged['country_name'] = merged['country_name_gdp']

    merged['country_name'] = merged['country_name'].astype(str).str.strip()

    # Features
    merged['log_gdp_pc'] = np.log1p(merged['gdp_per_capita'])
    merged['unemployment_next'] = merged.groupby('country_code')['unemployment_rate'].shift(-1)

    # Limpiar nulos para modelado
    df = merged.dropna(subset=['unemployment_rate', 'log_gdp_pc', 'unemployment_next']).copy()

    return df

# Panel lateral: rutas
st.sidebar.header("⚙️ Configuración de datos")
unemp_path = st.sidebar.text_input("Ruta dataset desempleo", "Unemployment_rate_dataset.csv")
gdp_path = st.sidebar.text_input("Ruta dataset PIB per cápita", "pib_per_capita_countries_dataset.csv")

# Cargar datos
try:
    df = load_and_prepare(unemp_path, gdp_path)
except Exception as e:
    st.error(f"Error al cargar/preparar los datos: {e}")
    st.stop()

# =============================
# Entrenamiento del modelo
# =============================
feat_cols = ['unemployment_rate', 'log_gdp_pc']
target = 'unemployment_next'

X = df[feat_cols].values
y = df[target].values
model = LinearRegression().fit(X, y)
b0, b1, b2 = float(model.intercept_), float(model.coef_[0]), float(model.coef_[1])

y_pred_all = model.predict(X)
r2 = r2_score(y, y_pred_all)
rmse = mean_squared_error(y, y_pred_all) ** 0.5
mae = np.mean(np.abs(y - y_pred_all))

# =============================
# UI principal
# =============================
st.title("📈 JobPulse — Análisis y Predicción del Desempleo")
st.caption("Explora cómo el PIB per cápita influye en la tasa de desempleo por país y simula escenarios.")

left, right = st.columns([2, 1])

with left:
    # País seleccionable
    countries = sorted(df['country_name'].dropna().unique().tolist())
    if not countries:
        st.error("No se encontraron países en el dataset. Verifica los archivos y rutas.")
        st.stop()

    country = st.selectbox("Selecciona un país", countries, index=min(10, len(countries)-1))
    sub = df[df['country_name'] == country].sort_values('year')

    if sub.empty:
        st.warning("No hay datos para el país seleccionado.")
    else:
        # Último registro disponible
        last = sub.iloc[-1]

        st.subheader(f"🇺🇳 {country} — Visión general")
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Año base", int(last['year']))
        with k2:
            st.metric("Desempleo actual (%)", f"{last['unemployment_rate']:.2f}")
        with k3:
            st.metric("PIB per cápita (US$)", f"{last['gdp_per_capita']:,.0f}")
        with k4:
            st.metric("R² Global", f"{r2:.3f}")

        # Slider de escenario de crecimiento
        growth_pct = st.slider("Crecimiento esperado del PIB (%) para el próximo año",
                               min_value=-10.0, max_value=15.0, value=2.0, step=0.5)

        # Predicción t+1 bajo escenario
        u_t = float(last['unemployment_rate'])
        gdp_t = float(last['gdp_per_capita'])
        gdp_next = gdp_t * (1.0 + growth_pct/100.0)
        log_gdp_next = float(np.log1p(gdp_next))
        pred_next = b0 + b1*u_t + b2*log_gdp_next

        st.markdown(f"**Predicción de desempleo (t+1)** con crecimiento del PIB de **{growth_pct:.1f}%**: "
                    f"**{pred_next:.2f}%**")

        # Gráfico dual: desempleo & PIB
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(sub['year'], sub['unemployment_rate'], label='Desempleo (%)')
        ax1.set_ylabel('Desempleo (%)')
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(sub['year'], sub['gdp_per_capita'], label='PIB per cápita (US$)')
        ax2.set_ylabel('PIB per cápita (US$)')
        ax1.set_xlabel('Año')
        ax1.set_title(f"Evolución — {country}")
        st.pyplot(fig)

        # Dispersión PIB vs desempleo
        st.subheader("Relación PIB per cápita vs Desempleo")
        fig2, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(df['gdp_per_capita'], df['unemployment_rate'], alpha=0.3)
        ax.set_xscale('log')
        ax.set_xlabel("PIB per cápita (US$) [log]")
        ax.set_ylabel("Desempleo (%)")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig2)

with right:
    st.subheader("📊 Métricas del modelo (global)")
    st.write(f"**R²:** {r2:.3f}")
    st.write(f"**RMSE:** {rmse:.3f}")
    st.write(f"**MAE:** {mae:.3f}")
    st.caption("Modelo: u_(t+1) = b0 + b1·u_t + b2·log(1 + PIB_pc_t)")

    with st.expander("🧪 Ecuación estimada y diagnóstico"):
        st.write(f"**b0 (intercepto):** {b0:.6f}")
        st.write(f"**b1 (coef u_t):** {b1:.6f}")
        st.write(f"**b2 (coef log PIB):** {b2:.6f}")

        # Predicción vs Real (dispersión)
        fig3, ax3 = plt.subplots(figsize=(5, 5))
        ax3.scatter(y, y_pred_all, alpha=0.5)
        mn, mx = float(min(y.min(), y_pred_all.min())), float(max(y.max(), y_pred_all.max()))
        ax3.plot([mn, mx], [mn, mx], linestyle='--')
        ax3.set_title("Predicho vs Real — Global")
        ax3.set_xlabel("Real (desempleo %)")
        ax3.set_ylabel("Predicho (desempleo %)")
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)

    with st.expander("🛠️ Depuración (columnas y muestra)"):
        st.write("Columnas del dataframe:", list(df.columns))
        st.dataframe(df.head(10))

st.success("Aplicación lista. Cambia el crecimiento del PIB para ver el efecto en tiempo real.")
