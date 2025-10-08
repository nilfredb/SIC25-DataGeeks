import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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
gdp_orgs = pd.read_csv('pib_per_capita_organizations_dataset.csv')
#caribbean = pd.read_csv('caribbean_countries.csv')

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