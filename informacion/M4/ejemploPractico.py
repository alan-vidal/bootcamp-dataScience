# Importamos las librerías necesarias para nuestro análisis
import pandas as pd  # Para la manipulación de datos (DataFrames)
import numpy as np   # Para operaciones numéricas (aunque pandas ya lo integra)
import seaborn as sns # Para visualizaciones estadísticas avanzadas
import matplotlib.pyplot as plt # Para crear y personalizar gráficos

# --- 1. Carga y Comprensión Inicial de los Datos ---
# En un proyecto real, cargaríamos un archivo (ej: pd.read_csv('iris.csv')).
# Aquí, por simplicidad, usamos la versión que viene con Seaborn.
# El objetivo es tener nuestros datos en una estructura tabular (DataFrame) para poder trabajar.
print("Cargando el conjunto de datos 'Iris'...")
df = sns.load_dataset('iris')

# Damos un primer vistazo a los datos para entender su estructura.
# .head() nos muestra las primeras 5 filas. Es vital para confirmar que los datos se cargaron correctamente.
print("\n--- Vistazo Inicial a los Datos ---")
print(df.head())

# .info() nos da un resumen técnico: tipos de variables (Dtype), cantidad de datos no nulos y uso de memoria.
# Esto nos ayuda a identificar inmediatamente si hay datos faltantes o si una variable numérica fue leída como texto (object).
print("\n--- Información General del DataFrame ---")
df.info()

# --- 2. Análisis Univariado: Explorando cada variable por separado ---
print("\n--- Análisis Univariado ---")

# a) Variable Categórica: 'species'
# Usamos .value_counts() para generar una tabla de frecuencia absoluta.
# Queremos saber cuántas muestras tenemos de cada especie de flor.
print("\nTabla de Frecuencia para la variable 'species':")
print(df['species'].value_counts())

# Visualización: Gráfico de barras para la frecuencia de cada especie.
# Un gráfico de barras es ideal para comparar las cantidades entre categorías.
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='species', palette='viridis')
plt.title('Distribución de Especies de Iris')
plt.xlabel('Especie')
plt.ylabel('Frecuencia Absoluta')
plt.show()

# b) Variable Cuantitativa: 'petal_length' (longitud del pétalo)
print("\nMedidas Estadísticas para 'petal_length':")
# Medidas de Tendencia Central
media_petalo = df['petal_length'].mean()
mediana_petalo = df['petal_length'].median()
moda_petalo = df['petal_length'].mode()[0] # .mode() devuelve una Serie, tomamos el primer elemento
print(f"Media: {media_petalo:.2f}")
print(f"Mediana: {mediana_petalo:.2f}")
print(f"Moda: {moda_petalo:.2f}")

# Medidas de Dispersión
rango_petalo = df['petal_length'].max() - df['petal_length'].min()
varianza_petalo = df['petal_length'].var() # Pandas usa la corrección de Bessel (n-1) por defecto
std_petalo = df['petal_length'].std()
print(f"Rango: {rango_petalo:.2f}")
print(f"Varianza (muestral): {varianza_petalo:.2f}")
print(f"Desviación Estándar: {std_petalo:.2f}")

# Visualización: Histograma para ver la distribución de la longitud del pétalo.
# El histograma nos permite observar la forma de la distribución (si es simétrica, sesgada, bimodal, etc.).
plt.figure(figsize=(8, 5))
sns.histplot(df['petal_length'], kde=True, color='purple') # kde=True añade una línea de densidad
plt.title('Distribución de la Longitud del Pétalo (petal_length)')
plt.xlabel('Longitud del Pétalo (cm)')
plt.ylabel('Frecuencia')
plt.show()

# Visualización: Gráfico de caja para identificar outliers y cuartiles.
# El boxplot resume la distribución: la caja representa el rango intercuartílico (IQR),
# la línea es la mediana (Q2) y los "bigotes" muestran el rango de los datos (excluyendo outliers).
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['petal_length'], color='lightblue')
plt.title('Boxplot de la Longitud del Pétalo')
plt.xlabel('Longitud del Pétalo (cm)')
plt.show()

# --- 3. Análisis Bivariado: Buscando relaciones entre dos variables ---
print("\n--- Análisis Bivariado ---")

# a) Relación entre dos variables cuantitativas: 'sepal_length' y 'sepal_width'
# Un gráfico de dispersión (scatter plot) es la mejor herramienta para visualizar esta relación.
# Buscamos patrones: ¿Aumenta una variable mientras la otra también lo hace (correlación positiva)? ¿O lo contrario (negativa)?
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='sepal_length', y='sepal_width', hue='species', palette='deep')
plt.title('Relación entre Longitud y Ancho del Sépalo')
plt.xlabel('Longitud del Sépalo (cm)')
plt.ylabel('Ancho del Sépalo (cm)')
plt.legend(title='Especie')
plt.show()

# b) Relación entre una variable cuantitativa y una categórica: 'petal_length' y 'species'
# Usamos boxplots para comparar la distribución de la longitud del pétalo entre las diferentes especies.
# Esto nos permite ver si la longitud del pétalo varía significativamente según la especie.
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='species', y='petal_length', palette='magma')
plt.title('Distribución de Longitud de Pétalo por Especie')
plt.xlabel('Especie')
plt.ylabel('Longitud del Pétalo (cm)')
plt.show()

# --- 4. Análisis Multivariado: Explorando interacciones complejas ---
print("\n--- Análisis Multivariado ---")

# a) Matriz de Correlación
# Calculamos la correlación de Pearson entre todas las variables numéricas.
# El resultado es una matriz que muestra qué tan fuerte es la relación lineal entre cada par de variables (-1 a 1).
# Seleccionamos solo las columnas numéricas para el cálculo.
df_numeric = df.select_dtypes(include=np.number)
matriz_corr = df_numeric.corr()
print("\nMatriz de Correlación:")
print(matriz_corr)

# Visualización: Mapa de calor (heatmap) para la matriz de correlación.
# El heatmap nos ayuda a identificar rápidamente las relaciones más fuertes (colores más intensos).
# annot=True muestra los valores numéricos en cada celda.
plt.figure(figsize=(10, 7))
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Mapa de Calor de la Matriz de Correlación')
plt.show()

# b) Gráfico de Dispersión Matricial (Pair Plot)
# Esta es una de las visualizaciones más potentes del EDA.
# Crea una matriz de gráficos: histogramas en la diagonal (análisis univariado)
# y gráficos de dispersión para cada par de variables (análisis bivariado).
# 'hue="species"' colorea los puntos según la especie, añadiendo una tercera dimensión al análisis.
print("\nGenerando Pair Plot (puede tardar unos segundos)...")
sns.pairplot(df, hue='species', palette='rocket')
plt.suptitle('Gráfico de Pares de las Variables de Iris', y=1.02) # y=1.02 para que el título no se superponga
plt.show()

print("\n--- Fin del Análisis Exploratorio de Datos ---")
