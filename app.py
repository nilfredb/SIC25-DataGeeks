
# app_streamlit_pro.py — JobPulse (UI Pro) — Interactivo y explicativo
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="JobPulse — Desempleo & PIB (Pro)", layout="wide")

# =============================
# Carga y preparación de datos
# =============================
@st.cache_data(show_spinner=True)
def load_and_prepare(unemp_path: str, gdp_path: str):
    unemp = pd.read_csv(unemp_path)
    gdp = pd.read_csv(gdp_path)
    # Unemployment wide -> long
    year_cols = unemp.columns[4:]
    unemp_long = unemp.melt(
        id_vars=["Country Name","Country Code","Indicator Name","Indicator Code"],
        value_vars=year_cols,
        var_name="year",
        value_name="unemployment_rate"
    ).dropna(subset=["unemployment_rate"])
    unemp_long["year"] = pd.to_numeric(unemp_long["year"], errors="coerce").astype("Int64")
    unemp_long.dropna(subset=["year"], inplace=True)
    unemp_long["year"] = unemp_long["year"].astype(int)
    unemp_long.rename(columns={"Country Name":"country_name","Country Code":"country_code"}, inplace=True)

    # GDP tidy
    gdp = gdp[["country_code","country_name","year","gdp_per_capita"]].copy()
    gdp["country_code"] = gdp["country_code"].astype(str).str.strip().str.upper()
    gdp["country_name"] = gdp["country_name"].astype(str).str.strip()
    gdp["year"] = pd.to_numeric(gdp["year"], errors="coerce").astype("Int64")
    gdp["gdp_per_capita"] = pd.to_numeric(gdp["gdp_per_capita"], errors="coerce")
    gdp.dropna(subset=["year"], inplace=True)
    gdp["year"] = gdp["year"].astype(int)

    gdp = gdp.rename(columns={"country_name":"country_name_gdp"})
    merged = pd.merge(
        unemp_long,
        gdp[["country_code","year","gdp_per_capita","country_name_gdp"]],
        on=["country_code","year"],
        how="inner"
    )
    if "country_name" in merged.columns:
        merged["country_name"] = merged["country_name"].fillna(merged["country_name_gdp"])
    else:
        merged["country_name"] = merged["country_name_gdp"]
    merged["country_name"] = merged["country_name"].astype(str).str.strip()

    merged["log_gdp_pc"] = np.log1p(merged["gdp_per_capita"])
    merged["unemployment_next"] = merged.groupby("country_code")["unemployment_rate"].shift(-1)

    # Tidy final
    merged = merged.dropna(subset=["unemployment_rate","gdp_per_capita"])
    return merged

@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame):
    dfm = df.dropna(subset=["unemployment_next"]).copy()
    X = dfm[["unemployment_rate","log_gdp_pc"]].values
    y = dfm["unemployment_next"].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    metrics = {
        "r2": float(r2_score(y, y_pred)),
        "rmse": float(mean_squared_error(y, y_pred) ** 0.5),
        "mae": float(np.mean(np.abs(y - y_pred))),
        "b0": float(model.intercept_),
        "b1": float(model.coef_[0]),
        "b2": float(model.coef_[1])
    }
    return model, metrics

def predict_next_unemp(b0, b1, b2, u_t, gdp_pc_next):
    return b0 + b1*u_t + b2*np.log1p(gdp_pc_next)

# =============================
# Sidebar — configuración
# =============================
st.sidebar.header("⚙️ Configuración de datos")
unemp_path = st.sidebar.text_input("Ruta dataset desempleo", "Unemployment_rate_dataset.csv")
gdp_path = st.sidebar.text_input("Ruta dataset PIB per cápita", "pib_per_capita_countries_dataset.csv")

df = load_and_prepare(unemp_path, gdp_path)
min_year, max_year = int(df["year"].min()), int(df["year"].max())

# Filtros globales
st.sidebar.header("🔎 Filtros")
years = st.sidebar.slider("Rango de años", min_year, max_year, (max(min_year, max_year-25), max_year))
region_countries = sorted(df["country_name"].unique().tolist())
default_country = "Dominican Republic" if "Dominican Republic" in region_countries else region_countries[0]
country = st.sidebar.selectbox("País principal", region_countries, index=region_countries.index(default_country))
compare_countries = st.sidebar.multiselect("Comparar con...", region_countries, default=["United States","Spain"] if len(region_countries) > 2 else [])

df_f = df[(df["year"]>=years[0]) & (df["year"]<=years[1])].copy()
sub = df_f[df_f["country_name"]==country].sort_values("year")

# =============================
# Header
# =============================
st.title("📈 JobPulse — Desempleo & PIB (Interactivo Pro)")
st.caption("Explora relaciones, compara países, simula escenarios y consulta métricas del modelo en una sola interfaz.")

# KPIs
k1,k2,k3,k4 = st.columns(4)
with k1:
    st.metric("Años filtrados", f"{years[0]}–{years[1]}")
with k2:
    st.metric("País principal", country)
with k3:
    st.metric("Observaciones filtradas", f"{len(df_f):,}")
with k4:
    corr = float(df_f["gdp_per_capita"].corr(df_f["unemployment_rate"]))
    st.metric("Correlación PIB↔Desempleo", f"{corr:.3f}")

# Train model once (global; sobre df con next)
model, metrics = train_model(df)
st.info(f"**Modelo t+1**: u_(t+1) = b0 + b1·u_t + b2·log(1+PIB_pc)\n\n"
        f"**R²**={metrics['r2']:.3f} | **RMSE**={metrics['rmse']:.3f} | **MAE**={metrics['mae']:.3f}")

# =============================
# Tabs
# =============================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔭 Exploración",
    "🧠 Modelo & Diagnóstico",
    "🧪 Simulador",
    "🆚 Comparar países",
    "📁 Datos & Descargas"
])

# ---- Exploración ----
with tab1:
    c1,c2 = st.columns([2,1])

    with c1:
        if sub.empty:
            st.warning("No hay datos para el país seleccionado en el rango de años.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sub["year"], y=sub["unemployment_rate"], mode="lines+markers",
                name="Desempleo (%)", yaxis="y1"
            ))
            fig.add_trace(go.Scatter(
                x=sub["year"], y=sub["gdp_per_capita"], mode="lines+markers",
                name="PIB per cápita (US$)", yaxis="y2"
            ))
            fig.update_layout(
                title=f"Evolución — {country}",
                xaxis_title="Año",
                yaxis=dict(title="Desempleo (%)", side="left"),
                yaxis2=dict(title="PIB per cápita (US$)", overlaying="y", side="right"),
                hovermode="x unified", height=420
            )
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Distribución de desempleo (hist + box)
        hfig = px.histogram(df_f[df_f["country_name"]==country], x="unemployment_rate", nbins=20, marginal="box",
                            title=f"Distribución desempleo — {country}")
        st.plotly_chart(hfig, use_container_width=True)

    # Dispersión global log-x
    sc = px.scatter(
        df_f, x="gdp_per_capita", y="unemployment_rate", color="country_name",
        title="Relación PIB per cápita vs Desempleo (log-x)",
        hover_data=["country_name","year"],
        labels={"gdp_per_capita":"PIB per cápita (US$)","unemployment_rate":"Desempleo (%)"},
        log_x=True, height=500
    )
    st.plotly_chart(sc, use_container_width=True)

# ---- Modelo ----
with tab2:
    left, right = st.columns([1,1])

    with left:
        # Real vs Predicho (global, sobre filas con next)
        dfm = df.dropna(subset=["unemployment_next"]).copy()
        X = dfm[["unemployment_rate","log_gdp_pc"]].values
        y = dfm["unemployment_next"].values
        y_hat = model.predict(X)
        comp = pd.DataFrame({"real":y, "pred":y_hat})
        comp["resid"] = comp["real"] - comp["pred"]

        rvsp = px.scatter(comp, x="real", y="pred", opacity=0.6,
                          title="Predicho vs Real (global)",
                          labels={"real":"Real (%)","pred":"Predicho (%)"})
        lim_min, lim_max = float(np.min([comp["real"].min(), comp["pred"].min()])), float(np.max([comp["real"].max(), comp["pred"].max()]))
        rvsp.add_shape(type="line", x0=lim_min, y0=lim_min, x1=lim_max, y1=lim_max, line=dict(dash="dash"))
        st.plotly_chart(rvsp, use_container_width=True)

    with right:
        # Residuos
        res_plot = px.scatter(comp.reset_index(), x=comp.index, y="resid", opacity=0.6,
                              title="Residuos (real - predicho)",
                              labels={"x":"Observación","resid":"Error"})
        res_plot.add_hline(y=0, line_dash="dash")
        st.plotly_chart(res_plot, use_container_width=True)

    st.subheader("Coeficientes del modelo")
    st.code(f"""
u_(t+1) = {metrics['b0']:.6f}
        + ({metrics['b1']:.6f})·u_t
        + ({metrics['b2']:.6f})·log(1 + PIB_pc_t)
""".strip(), language="text")

# ---- Simulador ----
with tab3:
    if sub.empty:
        st.warning("No hay datos para simular.")
    else:
        last = sub.iloc[-1]
        st.write(f"**País:** {country} | **Año base:** {int(last['year'])}")
        c1, c2, c3 = st.columns(3)
        with c1:
            u_t = float(last["unemployment_rate"])
            st.metric("Desempleo actual (%)", f"{u_t:.2f}")
        with c2:
            gdp_t = float(last["gdp_per_capita"])
            st.metric("PIB per cápita actual (US$)", f"{gdp_t:,.0f}")
        with c3:
            growth = st.slider("Crecimiento esperado del PIB (%) para t+1", -15.0, 20.0, 2.0, 0.5)
            gdp_next = gdp_t * (1 + growth/100.0)
            pred = predict_next_unemp(metrics["b0"], metrics["b1"], metrics["b2"], u_t, gdp_next)
            st.metric("Predicción desempleo (t+1) (%)", f"{pred:.2f}")

        # Curva de sensibilidad
        grid = np.linspace(-15, 20, 71)
        preds = [predict_next_unemp(metrics["b0"], metrics["b1"], metrics["b2"], u_t, gdp_t*(1+g/100.0)) for g in grid]
        sens = pd.DataFrame({"PIB crecimiento %":grid, "Desempleo t+1 (%)":preds})
        fig_sens = px.line(sens, x="PIB crecimiento %", y="Desempleo t+1 (%)",
                           title="Sensibilidad: efecto del crecimiento del PIB en u_(t+1)")
        fig_sens.add_vline(x=growth, line_dash="dash")
        st.plotly_chart(fig_sens, use_container_width=True)

# ---- Comparar países ----
with tab4:
    targets = [country] + [c for c in compare_countries if c != country]
    if not targets:
        st.info("Selecciona países en el panel lateral para comparar.")
    else:
        dcmp = df_f[df_f["country_name"].isin(targets)].copy()
        # Índice normalizado (base=100 en primer año del rango)
        def normalize(group):
            g = group.sort_values("year").copy()
            base_u = g["unemployment_rate"].iloc[0]
            base_g = g["gdp_per_capita"].iloc[0]
            g["u_idx"] = 100 * g["unemployment_rate"] / base_u if base_u != 0 else np.nan
            g["gdp_idx"] = 100 * g["gdp_per_capita"] / base_g if base_g != 0 else np.nan
            return g
        dcmp = dcmp.groupby("country_name", group_keys=False).apply(normalize)

        c1,c2 = st.columns(2)
        with c1:
            fig_u = px.line(dcmp, x="year", y="u_idx", color="country_name",
                            title="Índice de desempleo (base=100 en primer año del rango)")
            st.plotly_chart(fig_u, use_container_width=True)
        with c2:
            fig_g = px.line(dcmp, x="year", y="gdp_idx", color="country_name",
                            title="Índice de PIB per cápita (base=100 en primer año del rango)")
            st.plotly_chart(fig_g, use_container_width=True)

        # Mapa de calor de desempleo por año
        heat = dcmp.pivot_table(index="country_name", columns="year", values="unemployment_rate", aggfunc="mean")
        heat = heat.reindex(index=targets)
        heat_fig = px.imshow(heat, aspect="auto", color_continuous_scale="Viridis",
                             title="Mapa de calor — Desempleo (%)")
        st.plotly_chart(heat_fig, use_container_width=True)

# ---- Datos & Descargas ----
with tab5:
    st.subheader("Datos filtrados")
    st.dataframe(df_f.sort_values(["country_name","year"]).reset_index(drop=True), use_container_width=True, height=420)
    csv = df_f.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar CSV filtrado", data=csv, file_name="jobpulse_filtrado.csv", mime="text/csv")

    st.subheader("Documentación rápida")
    st.markdown("""
- **Desempleo**: tasa reportada por país y año.
- **PIB per cápita**: dólares constantes (según fuente).
- **log_gdp_pc**: transformación log(1 + PIB_pc) para estabilizar escala.
- **unemployment_next**: objetivo del modelo = desempleo del próximo año (shift -1 por país).
""")

st.caption("© JobPulse — Visual analytics & forecasting")
