# 🌍 GDP vs Unemployment Analysis (1991–2023)

Este proyecto analiza la **relación entre el PIB per cápita y la tasa de desempleo** a nivel mundial entre los años **1991 y 2023**, utilizando datos provenientes del **Banco Mundial (World Bank)**.  
El objetivo es observar tendencias económicas, correlaciones y diferencias entre países a lo largo del tiempo mediante herramientas de análisis de datos y visualización interactiva.

---

## 📊 Descripción general

El análisis se centra en responder la pregunta:
> **¿Existe una relación estadísticamente significativa entre el crecimiento económico (PIB per cápita) y la tasa de desempleo?**

Para ello se realiza:
1. Limpieza y normalización de los datos.
2. Fusión de datasets de PIB y desempleo por país y año.
3. Cálculo de correlaciones y tendencias.
4. Visualización interactiva mediante un dashboard.

---

## 🧠 Tecnologías utilizadas

| Tipo de tecnología | Herramienta | Descripción |
|--------------------|-------------|-------------|
| **Lenguaje de programación** | 🐍 **Python** | Base del proyecto y análisis de datos. |
| **Librerías principales** | **pandas**, **NumPy** | Limpieza, manipulación y cálculo estadístico. |
| **Visualización** | **Matplotlib** | Gráficos de correlación y evolución temporal. |
| **Dashboard interactivo** | **Streamlit** | Interfaz web para explorar los resultados. |
| **Fuentes de datos** | **CSV (World Bank Data)** | Datos de PIB per cápita y desempleo por país. |
| **Proceso ETL** | — | Extracción, transformación y carga de datos. |
| **Entorno de desarrollo** | **VS Code / Jupyter Notebook** | Codificación, pruebas y visualización. |
| **Control de versiones** | **Git / GitHub** | Versionado del código y documentación. |

---

## ⚙️ Flujo de trabajo (ETL)

1. **Extracción:**  
   Se importan los archivos CSV descargados del Banco Mundial.

2. **Transformación:**  
   - Conversión de formato *wide → long* para unificar columnas de años.  
   - Limpieza de valores nulos y duplicados.  
   - Fusión de datasets por `country_code` y `year`.

3. **Carga:**  
   - Almacenamiento de los datos procesados en archivos CSV limpios.  
   - Visualización mediante gráficos y dashboard interactivo.

---

## 📈 Ejemplo de análisis

- Cálculo de correlación entre PIB per cápita y desempleo:  
  ```python
  df['GDP_per_capita'].corr(df['Unemployment_rate'])
  ```
- Visualización de tendencias:
  ```python
  import matplotlib.pyplot as plt
  plt.scatter(df['GDP_per_capita'], df['Unemployment_rate'])
  plt.title("PIB vs Desempleo")
  plt.xlabel("PIB per cápita (USD)")
  plt.ylabel("Tasa de desempleo (%)")
  plt.show()
  ```

---

## 🚀 Ejecución del proyecto

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/tuusuario/gdp-unemployment-analysis.git
   cd gdp-unemployment-analysis
   ```

2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Ejecutar el dashboard:
   ```bash
   streamlit run app.py
   ```

---

## 🧾 Resultados esperados

- Visualización clara de la relación entre desarrollo económico y desempleo.
- Cálculo de correlaciones por país y año.
- Dashboard interactivo para exploración libre.
- Base sólida para futuras predicciones o análisis econométricos.

---

## 📚 Créditos

Proyecto desarrollado por **DataGeeks** para el Samsung Innovation Campus 2025  
📍 República Dominicana  
