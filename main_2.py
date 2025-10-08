import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Guardar outputs
OUT_DIR = Path('outputs')
OUT_DIR.mkdir(exist_ok=True)
def savefig_safe(name: str):
    plt.tight_layout()
    plt.savefig(OUT_DIR / name, dpi=120)
    plt.show()

# Cargar los datasets

unemp = pd.read_csv('Unemployment_rate_dataset.csv')
gdp_countries = pd.read_csv('pib_per_capita_countries_dataset.csv')

# Preparar el dataset de desempleo (de ancho a largo)
year_cols = unemp.columns[4:]
unemp_long = unemp.melt(
    id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
    value_vars=year_cols,
    var_name='year',
    value_name='unemployment_rate'
    ).dropna(subset=['unemployment_rate'])

unemp_long['year'] = unemp_long['year'].astype(int)


# Renombrar columnas para claridad
unemp_long = unemp_long.rename(columns={
    'Country Name': 'country_name',
    'Country Code': 'country_code'
})

print("Dataset de desempleo listo:", unemp_long.shape)
print(unemp_long.head())


# Preparar el dataset de PIB per capita (paises)

gdp_countries = gdp_countries[['country_code', 'country_name', 'year', 'gdp_per_capita']].copy()
gdp_countries['country_code'] = gdp_countries['country_code'].astype(str).str.strip().str.upper()
gdp_countries['country_name'] = gdp_countries['country_name'].astype(str).str.strip()
gdp_countries['year'] = pd.to_numeric(gdp_countries['year'], errors='coerce').astype('Int64')
gdp_countries['gdp_per_capita'] = pd.to_numeric(gdp_countries['gdp_per_capita'], errors='coerce')


# Filtrar filas con año valido
gdp_countries = gdp_countries.dropna(subset=['year']).copy()
gdp_countries['year'] = gdp_countries['year'].astype(int)

print("Dataset de PIB per capita (paises) listo:", gdp_countries.shape)
print(gdp_countries.head())

# Renombre de columnas para claridad
# Hacer merge de desempleo con PIB per capita (paises)
gdp_countries_sel = gdp_countries.rename(columns={'country_name': 'country_name_gdp'})

merged = pd.merge(
    unemp_long,
    gdp_countries_sel[['country_code', 'year', 'gdp_per_capita', 'country_name_gdp']],
    on=['country_code', 'year'],
    how='inner'
)

print(merged.head())

# Crear una sola columna 'country_name' priorizando la del desempleo

if 'country_name' in merged.columns:
    merged['country_name'] = merged['country_name'].fillna(merged['country_name_gdp'])
else:
    merged['country_name'] = merged['country_name_gdp']

merged['country_name'] = merged['country_name'].astype(str).str.strip()

print("Datos combinados:", merged.shape)
print(merged[['country_name','country_code','year','unemployment_rate','gdp_per_capita']].head())
print("\nColumnas merged:", merged.columns.tolist())

# Estudio de la relacion entre desempleo y PIB per capita
# Correlación global
corr = merged['gdp_per_capita'].corr(merged['unemployment_rate'])
print(f"\nCorrelación PIB per cápita vs Desempleo: {corr:.3f}")

# Promedio de desempleo por pais
avg_unemp = (merged.groupby('country_name')['unemployment_rate']
             .mean()
             .sort_values(ascending=False))
print("\nPaíses con mayor desempleo promedio:")
print(avg_unemp.head(10))

# Promedio por año (util para ver picos o caidas globales)
avg_per_year = merged.groupby('year')['unemployment_rate'].mean()
print(f"\nAño con mayor desempleo: {avg_per_year.idxmax()} ({avg_per_year.max():.2f}%)")
print(f"Año con menor desempleo: {avg_per_year.idxmin()} ({avg_per_year.min():.2f}%)")

# Graficar la evolución del desempleo en algunos países seleccionados
# a) Línea de desempleo para varios países
paises = ['Dominican Republic', 'United States', 'Japan', 'South Africa', 'Spain']

plt.figure(figsize=(10,6))
for pais in paises:
    datos = merged[merged['country_name'] == pais].sort_values('year')
    if not datos.empty:
        plt.plot(datos['year'], datos['unemployment_rate'], label=pais)
plt.title('Evolución del desempleo (1991–2023)')
plt.xlabel('Año')
plt.ylabel('Tasa de desempleo (%)')
plt.legend()
plt.grid(True)
savefig_safe('lineas_desempleo.png')

# b) Dispersión PIB per cápita vs desempleo (escala log en X)
plt.figure(figsize=(8,6))
plt.scatter(merged['gdp_per_capita'], merged['unemployment_rate'], alpha=0.6)
plt.title('Relación entre PIB per cápita y Desempleo')
plt.xlabel('PIB per cápita (US$)')
plt.ylabel('Desempleo (%)')
plt.xscale('log')
plt.grid(True)
savefig_safe('okun_ley.png')

# Grafico doble eje

pais = 'Dominican Republic'  # puedes cambiarlo por 'DOM' si prefieres filtrar por código
serie = merged[merged['country_name'] == pais].sort_values('year')

if not serie.empty:
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(serie['year'], serie['unemployment_rate'], color='green', label='Desempleo (%)')
    ax1.set_ylabel('Desempleo (%)', color='green')

    ax2 = ax1.twinx()
    ax2.plot(serie['year'], serie['gdp_per_capita'], color='blue', label='PIB per cápita (US$)')
    ax2.set_ylabel('PIB per cápita (US$)', color='blue')

    plt.title(f'Evolución del PIB y Desempleo — {pais}')
    plt.xlabel('Año')
    plt.grid(True)
    savefig_safe(f'dual_DOM.png')
else:
    print(f"No hay datos para {pais}")

# Conclusion final

if corr < 0:
    conclusion = "Relación inversa: cuando el PIB per cápita sube, el desempleo tiende a bajar."
elif corr > 0:
    conclusion = "Relación directa: el desempleo aumenta al subir el PIB per cápita."
else:
    conclusion = "No se observa relación clara."

print("\nConclusión general:")
print(conclusion)

base = (merged[['country_code','country_name','year','unemployment_rate','gdp_per_capita']]
        .dropna()
        .sort_values(['country_code','year'])
        .copy())

# (opcional) Log para suavizar escala del PIB per cápita
base['log_gdp_pc'] = np.log1p(base['gdp_per_capita'])

# Target: desempleo del próximo año por país (shift -1)
base['unemployment_next'] = base.groupby('country_code')['unemployment_rate'].shift(-1)

# Features sencillas (nivel 5/10): desempleo actual y nivel de PIB per cápita (log)
feat_cols = ['unemployment_rate', 'log_gdp_pc']
target_col = 'unemployment_next'
df_simple = base.dropna(subset=feat_cols + [target_col]).copy()

# 2) Split temporal (evita fuga de info). Test = últimos 2 años globales
years = np.sort(df_simple['year'].unique())
if len(years) <= 3:
    cutoff = years[int(len(years)*0.7)]
    train = df_simple[df_simple['year'] <= cutoff]
    test  = df_simple[df_simple['year'] > cutoff]
else:
    last2 = years[-2]
    train = df_simple[df_simple['year'] < last2]
    test  = df_simple[df_simple['year'] >= last2]

X_train = train[feat_cols].values
y_train = train[target_col].values
X_test  = test[feat_cols].values
y_test  = test[target_col].values

# 3) Modelo lineal simple
lin = LinearRegression()
lin.fit(X_train, y_train)

# 4) Métricas
y_hat_tr = lin.predict(X_train)
y_hat_te = lin.predict(X_test)

def _scores(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return r2, mae, rmse

r2_tr, mae_tr, rmse_tr = _scores(y_train, y_hat_tr)
r2_te, mae_te, rmse_te = _scores(y_test,  y_hat_te)

print("\n== JobPulse (simple t+1) :: Métricas ==")
print(f"Train | R²={r2_tr:.3f} | MAE={mae_tr:.3f} | RMSE={rmse_tr:.3f}")
print(f"Test  | R²={r2_te:.3f} | MAE={mae_te:.3f} | RMSE={rmse_te:.3f}")

# 5) Ecuación del modelo
print("\nEcuación (t+1):")
print(f"u_(t+1) = {lin.intercept_:.4f}"
      f" + ({lin.coef_[0]:.6f})*u_t"
      f" + ({lin.coef_[1]:.6f})*log(1+PIB_pc_t)")

# 6) Gráficos simples
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_hat_te, alpha=0.6)
mn = float(min(y_test.min(), y_hat_te.min()))
mx = float(max(y_test.max(), y_hat_te.max()))
plt.plot([mn, mx], [mn, mx])
plt.title('Predicción vs Real — Test (Lineal t+1)')
plt.xlabel('Real (desempleo %)')
plt.ylabel('Predicho (desempleo %)')
plt.grid(True)
savefig_safe('jobpulse_t1_pred_vs_real.png')

resid = y_test - y_hat_te
plt.figure(figsize=(8,4))
plt.scatter(np.arange(len(resid)), resid, alpha=0.6)
plt.axhline(0)
plt.title('Residuos — Test')
plt.xlabel('Observación')
plt.ylabel('Error (real - predicho)')
plt.grid(True)
savefig_safe('jobpulse_t1_residuos.png')

# 7) Comparación real vs predicho (muestra)
comparison = test[['country_name','year','unemployment_rate', 'unemployment_next']].copy()
comparison['pred_next'] = y_hat_te
comparison = comparison.rename(columns={'unemployment_rate':'u_t',
                                        'unemployment_next':'u_t1_real'})
print("\nComparación (muestra):")
print(comparison.head(10))

# 8) Simulador sencillo: “¿y si el PIB per cápita sube X%?”
#    Asume que u_(t+1) ≈ b0 + b1*u_t + b2*log(1 + PIB_pc_(t+1)),
#    con PIB_pc_(t+1) ≈ PIB_pc_t * (1 + x/100). u_t se mantiene igual (escenario naïve).
b0 = float(lin.intercept_)
b1 = float(lin.coef_[0])
b2 = float(lin.coef_[1])

def escenario_pib(country: str, growth_pct_next: float):
    sub = df_simple[df_simple['country_name'] == country].sort_values('year')
    if sub.empty:
        raise ValueError(f"No hay datos para {country}")
    last = sub.iloc[-1]
    u_t = float(last['unemployment_rate'])
    gdp_t = float(last['gdp_per_capita'])
    gdp_next = gdp_t * (1.0 + growth_pct_next/100.0)
    log_gdp_next = np.log1p(gdp_next)
    y_pred_next = b0 + b1*u_t + b2*log_gdp_next
    return {
        'country': country,
        'last_year': int(last['year']),
        'assumed_gdp_pc_growth_%': growth_pct_next,
        'u_t_used': u_t,
        'gdp_pc_t': gdp_t,
        'gdp_pc_next_est': gdp_next,
        'predicted_unemployment_next_%': float(y_pred_next)
    }

print("\n== Escenarios simples ==")
for pais, g in [('Dominican Republic', 2.0), ('Spain', -1.0), ('United States', 1.5)]:
    try:
        print(escenario_pib(pais, g))
    except Exception as e:
        print(f"{pais}: {e}")