## 🔹 1. **Características principales de Seaborn**

### 📌 Definición:
**Seaborn** es una librería de visualización de datos basada en **Matplotlib** que proporciona una interfaz de alto nivel para crear gráficos estadísticos atractivos y profesionales. Está diseñada para trabajar directamente con estructuras de datos como **Pandas DataFrames**.

### ✅ Características principales:

- Interfaz intuitiva y fácil de usar.
- Estilos predefinidos para gráficos más bonitos sin necesidad de ajustar estilos manualmente.
- Integración directa con **Pandas**.
- Soporte avanzado para visualizar distribuciones, relaciones entre variables y análisis categóricos.
- Capacidad de generar **gráficos complejos** (como heatmaps, pairplots, facet grids) con pocas líneas de código.

### 🧪 Ejemplo básico:
```PYTHON
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar un dataset de ejemplo
tips = sns.load_dataset("tips")

# Gráfico de barras simple
sns.barplot(x="day", y="total_bill", data=tips)
plt.show()

```

### 💡 Utilidad:
Ideal para explorar datos de forma visual, detectar patrones, tendencias o anomalías en grandes volúmenes de información.

---

## 🔹 2. **Gráficos de Distribución**

### 📌 Definición:
Los gráficos de distribución muestran cómo se distribuyen los valores de una variable en el conjunto de datos. Ayudan a identificar patrones como la centralidad, dispersión o formas de las distribuciones (normal, sesgada, etc.).

---

### 📊 2.1 `histplot`

### 📌 Definición:
Muestra la distribución de datos numéricos mediante **histogramas**, dividiendo los datos en **bins (intervalos)** y mostrando la frecuencia de cada uno.

### 🧪 Ejemplo:
```python
sns.histplot(data=tips, x="total_bill", bins=20, kde=False)
plt.title("Distribución del Total de Factura")
plt.show()
```

### 💡 Utilidad:
- Útil para ver la forma general de la distribución.
- Permite ajustar el número de bins para explorar detalles o agrupaciones.

---

### 📊 2.2 `kdeplot`

### 📌 Definición:
Muestra la **densidad de probabilidad estimada (KDE)** de una variable, lo cual permite tener una visión más suave y continua de la distribución.

### 🧪 Ejemplo:
```python
sns.kdeplot(data=tips, x="total_bill", fill=True)
plt.title("Estimación de Densidad del Total de Factura")
plt.show()
```

### 💡 Utilidad:
- Muy útil cuando se quiere ver la forma de la distribución sin depender de los bins.
- Ideal para comparar distribuciones entre grupos.

---

## 🔹 3. **¿Cuándo usar `histplot` y cuándo usar `kdeplot`?**

| Gráfico     | ¿Cuándo usarlo? |
|-------------|------------------|
| `histplot`  | Cuando quieras mostrar la frecuencia real de los datos divididos en intervalos. Es útil para ver la cantidad de observaciones por rango. |
| `kdeplot`   | Cuando quieras una representación suave de la distribución de los datos, especialmente útil para comparar varias distribuciones o cuando los datos son continuos. |

---

## 🔹 4. **Gráficos de Dispersión y Correlación**

### 📌 Definición:
Sirven para visualizar la relación entre dos variables numéricas.

---

### 📊 `scatterplot`

### 🧪 Ejemplo:
```python
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="day")
plt.title("Relación entre Factura y Propina")
plt.show()
```

### 💡 Utilidad:
- Identifica correlaciones positivas/negativas.
- Detecta outliers o patrones específicos.

---

### 📊 `jointplot`

### 🧪 Ejemplo:
```python
sns.jointplot(data=tips, x="total_bill", y="tip", kind="reg")
plt.suptitle("Distribución conjunta de Factura y Propina")
plt.tight_layout()
plt.show()
```

### 💡 Utilidad:
Combina scatterplot + histogramas marginales para dar una visión completa de la relación y distribución individual de las variables.

---

## 🔹 5. **Gráficos de Variables Categóricas**

### 📌 Definición:
Son gráficos usados para visualizar la relación entre una variable categórica y otra numérica.

---

### 📊 Ejemplos comunes:

#### 📈 `barplot`
```python
sns.barplot(x="day", y="total_bill", data=tips)
plt.title("Promedio de Factura por Día")
plt.show()
```

#### 📉 `boxplot`
```python
sns.boxplot(x="day", y="total_bill", data=tips)
plt.title("Distribución de Facturas por Día")
plt.show()
```

#### 📊 `violinplot`
```python
sns.violinplot(x="day", y="total_bill", data=tips)
plt.title("Distribución de Facturas por Día (Violín)")
plt.show()
```

### 💡 Utilidad:
- Comparar grupos o categorías.
- Mostrar medidas centrales, dispersión y formas de distribución por categoría.

---

## 🔹 6. **Heatmap**

### 📌 Definición:
Un heatmap muestra datos en una matriz donde cada celda está coloreada según el valor correspondiente. Es ideal para visualizar matrices de correlación o tablas de contingencia.

### 🧪 Ejemplo:
```PYTHON
import numpy as np

# Matriz de correlación
corr = tips.select_dtypes(include=np.number).corr()

# Heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Mapa de Calor de Correlación")
plt.show()

```

### 💡 Utilidad:
- Visualizar correlaciones entre múltiples variables.
- Identificar patrones en matrices de datos complejas.

---

## 🔹 7. **Grillas de Gráficos (`FacetGrid`, `PairGrid`, `JointGrid`)**

### 📌 Definición:
Permiten crear **grids (cuadrículas)** de gráficos basadas en niveles de una o más variables categóricas.

---

### 📊 `FacetGrid`

### 🧪 Ejemplo:
```PYTHON
g = sns.FacetGrid(tips, col="time", row="smoker")
g.map(sns.scatterplot, "total_bill", "tip")
plt.show()

```

### 💡 Utilidad:
- Dividir los datos en subconjuntos basados en variables categóricas.
- Aplicar el mismo tipo de gráfico a cada subconjunto.

---

### 📊 `PairGrid`

### 🧪 Ejemplo:
```PYTHON
iris = sns.load_dataset("iris")
g = sns.PairGrid(iris, hue="species")
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
g.add_legend()
plt.show()

```

### 💡 Utilidad:
- Explorar relaciones entre todas las combinaciones posibles de variables numéricas.
- Identificar patrones o clusters por categorías.

---

## 🔹 8. **Comparación entre `PairGrid` y `FacetGrid`**

| Característica         | `FacetGrid`                              | `PairGrid`                                  |
|------------------------|-------------------------------------------|----------------------------------------------|
| **Propósito**          | Dividir datos por variables categóricas    | Comparar pares de variables numéricas        |
| **Uso típico**         | Scatterplots, lineplots, barplots por grupo | Análisis multivariante                       |
| **Ejemplo de uso**     | Analizar ventas por región y año           | Comparar largo vs ancho de sépalos en flores |
| **Flexibilidad**       | Menos flexible                            | Más flexible (permite distintos gráficos)    |
| **Visualización**      | Una variable por eje + facetas             | Todas las combinaciones de variables         |

---

## 📌 Conclusión

**Seaborn** es una herramienta poderosa y versátil para el análisis exploratorio de datos. Ofrece una amplia gama de gráficos listos para usar, desde simples histogramas hasta mapas de calor y grillas complejas. Su integración con Pandas facilita enormemente el trabajo con datos reales.
