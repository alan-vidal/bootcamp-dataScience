### El Flujo del Análisis de Datos

Imagina que recibes un conjunto de datos nuevo. Tu primer objetivo no es aplicar modelos complejos, sino entender la naturaleza de tus datos. Este proceso es un viaje que va de lo simple a lo complejo, y se conoce como **Análisis Exploratorio de Datos (EDA)**.

---

### 1. ¿Qué es el Análisis Exploratorio de Datos (EDA)? 🕵️‍♀️

El **Análisis Exploratorio de Datos** es el primer paso en cualquier proyecto de ciencia de datos. Es el proceso de investigar, resumir y visualizar un conjunto de datos para descubrir patrones, identificar anomalías (como outliers), probar hipótesis y verificar suposiciones con la ayuda de estadísticas resumidas y representaciones gráficas. Piénsalo como el trabajo de un detective: antes de acusar a un sospechoso, recopilas y examinas toda la evidencia.

El EDA no se trata de tener respuestas definitivas, sino de formular mejores preguntas.

---

### 2. Tipos de Variables: La Materia Prima

Antes de analizar, debemos conocer los "ingredientes" de nuestros datos: las variables. Se dividen en dos grandes familias:

- #### **Variables Categóricas (Cualitativas)**

    Representan cualidades o categorías.

    - **Nominales**: No tienen un orden intrínseco.

        - **Ejemplo**: `País` ('Chile', 'Argentina', 'Perú'), `Género` ('Masculino', 'Femenino').

    - **Ordinales**: Sí tienen un orden o jerarquía.

        - **Ejemplo**: `Nivel Educativo` ('Básico', 'Medio', 'Superior'), `Calificación` ('Malo', 'Regular', 'Bueno').

- #### **Variables Cuantitativas (Numéricas)**

    Representan cantidades medibles.

    - **Discretas**: Toman valores enteros y contables.

        - **Ejemplo**: `Número de hijos` (0, 1, 2, 3), `Cantidad de productos en un carrito`.

    - **Continuas**: Pueden tomar cualquier valor dentro de un rango.

        - **Ejemplo**: `Altura` (1.75m, 1.68m), `Temperatura` (25.5°C).


---

### 3. Niveles de Análisis: De lo Simple a lo Complejo

Una vez que conocemos nuestras variables, comenzamos a analizarlas en diferentes niveles.

#### **Análisis Univariado: Mirando una Sola Variable**

Nos enfocamos en una única variable a la vez para describirla y entender su comportamiento.

- **Para variables categóricas**: Usamos **tablas de frecuencia**.

    - **Tabla de Frecuencia**: Muestra cuántas veces aparece cada categoría.

        - **Absoluta**: El conteo directo. (`Ej: 20 personas de Chile`).

        - **Relativa**: El porcentaje sobre el total. (`Ej: 20 de 100 personas son de Chile, 20%`).

        - **Acumulada**: La suma de las frecuencias relativas. (`Ej: Frecuencia de 'Bueno' + 'Regular'`).

- **Para variables cuantitativas**: Usamos medidas estadísticas.

    - **Medidas de Tendencia Central**: Buscan el "centro" de los datos.

        - **Media (μ, xˉ)**: El promedio aritmético. Sensible a outliers.

        - **Mediana**: El valor que está justo en el medio de los datos ordenados. Robusta a outliers.

        - **Moda**: El valor que más se repite. Puede haber más de una.

    - **Medidas de Dispersión**: Miden qué tan "esparcidos" están los datos.

        - **Rango**: Diferencia entre el valor máximo y mínimo.

        - **Varianza (σ2, s2)**: El promedio de las distancias al cuadrado desde cada punto hasta la media.

        - **Desviación Estándar (σ, s)**: La raíz cuadrada de la varianza. Se interpreta en las mismas unidades que los datos originales, lo que la hace más intuitiva.

    - **Medidas de Posición**: Nos dicen dónde se encuentra un dato en relación con los demás.

        - **Cuartiles**: Dividen los datos en 4 partes iguales (Q1, Q2=Mediana, Q3).

        - **Deciles**: Dividen los datos en 10 partes iguales.

        - **Percentiles**: Dividen los datos en 100 partes iguales. El percentil 80 (P80) es el valor bajo el cual se encuentra el 80% de los datos.


---

#### **Población vs. Muestra y la Corrección de Bessel**

Es crucial entender esta diferencia:

- **Población**: Es el **conjunto completo** de todos los individuos u objetos de interés. Calcular la varianza poblacional (σ2) es directo si tienes todos los datos.

- **Muestra**: Es un **subconjunto** de la población que utilizamos para inferir características de la población total.


Cuando calculamos la varianza de una muestra (s2), tendemos a subestimar la varianza real de la población. La **Corrección de Bessel** ajusta esto dividiendo por n−1 en lugar de n (el tamaño de la muestra). Esto nos da una estimación más precisa (insesgada) de la varianza poblacional.


$$Varianza Muestral (Corregida): s^2 ={\sum_{i=1}^n {(x_i - \bar{x})^2} \over {(n - 1)}} $$
---

#### **Puntos Atípicos o Outliers** outlier

Son observaciones que se desvían significativamente del resto de los datos. Pueden ser errores de medición o valores genuinamente extremos. Identificarlos es clave, ya que pueden distorsionar medidas como la media y la varianza. Una técnica común para detectarlos es usar el rango intercuartílico (IQR = Q3 - Q1). Se considera un outlier un punto que está por debajo de Q1−1.5×IQR o por encima de Q3+1.5×IQR.

---

#### **Análisis Bivariado: Relacionando Dos Variables**

Aquí exploramos la relación entre **dos variables** simultáneamente.

- **Categórica y Categórica**: Tablas de contingencia (o tablas cruzadas).

- **Cuantitativa y Cuantitativa**: Gráficos de dispersión (Scatter Plot) para ver la forma, dirección y fuerza de la relación. Calculamos la **correlación**.

- **Categórica y Cuantitativa**: Gráficos de cajas (Box Plots) por categoría o histogramas/densidades separadas.


---

#### **Análisis Multivariado: Más de Dos Variables a la Vez**

Este análisis examina la relación simultánea entre **tres o más variables**. Es aquí donde se revelan las interacciones más complejas.

- **Técnicas de Análisis Multivariable**:

    - **Correlación**: La matriz de correlación nos muestra el coeficiente de correlación (ej., Pearson) entre cada par de variables cuantitativas. Es una extensión del análisis bivariado a múltiples variables.

        - _Ejemplo en Python_: `datos.corr()` usaría pandas para calcular la matriz. Un mapa de calor (heatmap) de `seaborn` es ideal para visualizarla.

    - **Análisis de Componentes Principales (PCA)**: Es una técnica de **reducción de dimensionalidad**. Transforma un gran número de variables correlacionadas en un número menor de nuevas variables no correlacionadas llamadas "componentes principales", reteniendo la mayor cantidad de información (varianza) posible.

        - _Ejemplo_: Si tienes 10 variables sobre las características de un vino, PCA podría ayudarte a resumirlas en 2 o 3 componentes principales que expliquen, por ejemplo, la "calidad general" y el "perfil de acidez".

    - **Gráficos de Dispersión Matricial (Pair Plot)**: Crea una matriz de gráficos donde las celdas de la diagonal principal muestran la distribución de cada variable (histograma o densidad) y las celdas fuera de la diagonal muestran gráficos de dispersión entre cada par de variables.

        - _Ejemplo en Python_: `seaborn.pairplot(datos)` es una forma rápida y poderosa de obtener una visión general de las relaciones en tu dataset.


---

### 4. Visualización de Datos: Contando la Historia 📊

La visualización es fundamental en todo el proceso. Un buen gráfico comunica patrones complejos de manera rápida y efectiva. "Una imagen vale más que mil filas de una tabla".

#### **Tipos de Visualización de Datos**

- **Para Análisis Univariado**:

    - **Histograma**: Muestra la distribución de una variable cuantitativa. Ideal para ver la forma (simétrica, sesgada).

    - **Gráfico de Cajas (Box Plot)**: Resume la distribución de datos cuantitativos mostrando la mediana, cuartiles y outliers. Excelente para comparar distribuciones entre categorías.

    - **Gráfico de Barras**: Compara cantidades para diferentes categorías.

    - **Gráfico de Torta o Anillo (Pie/Doughnut Chart)**: Muestra la proporción de cada categoría. Usar con moderación, generalmente un gráfico de barras es más claro.

- **Para Análisis Bivariado y Multivariado**:

    - **Gráfico de Dispersión (Scatter Plot)**: Muestra la relación entre dos variables cuantitativas.

    - **Mapa de Calor (Heatmap)**: Visualiza una matriz (como la de correlación) usando colores para representar valores.

    - **Gráfico de Líneas**: Muestra la evolución de una variable cuantitativa a lo largo del tiempo o de otra variable continua.

    - **Pair Plot**: Como se mencionó, para una vista rápida multivariada.


Este flujo, desde entender las variables hasta analizar sus interacciones complejas, es la base de un análisis de datos robusto y consciente.
