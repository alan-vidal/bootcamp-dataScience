## 🧮 1. **Regresión Lineal**

### 🔹 Definición:
La regresión lineal es una técnica estadística que modela la relación entre una variable dependiente (Y) y una o más variables independientes (X), asumiendo una relación lineal entre ellas.

### ✅ Utilidad:
- **Predecir valores futuros**: Por ejemplo, predecir el precio de un inmueble basado en su tamaño.
- **Explicar fenómenos**: Entender cómo influye la publicidad en las ventas.
- **Modelo base**: Sirve como punto de partida antes de aplicar modelos más complejos (como árboles de decisión o redes neuronales).
- **Interpretable**: Permite entender claramente cómo afectan las variables al resultado.

### 📊 Ejemplo numérico:

| Años de Experiencia (X) | Salario Anual (en miles) (Y) |
|--------------------------|-------------------------------|
| 2                        | 50                            |
| 4                        | 60                            |
| 6                        | 70                            |
| 8                        | 80                            |
| 10                       | 90                            |

Ecuación del modelo ajustado:
$$
\text{Salario} = 40 + 5 \cdot (\text{Años de experiencia})
$$

- **Intercepto (β₀)**: Salario inicial sin experiencia → 40
- **Pendiente (β₁)**: Cada año adicional aporta $5,000 → 5

---

## 📈 2. **Método de Mínimos Cuadrados**

### 🔹 Definición:
Es un método matemático utilizado para encontrar los coeficientes de una recta que mejor se ajuste a los datos, minimizando la suma de los cuadrados de los errores residuales.

### ✅ Utilidad:
- Garantiza la **mejor línea de ajuste posible** dada la data.
- Es la base de muchos modelos paramétricos.
- Tiene solución analítica clara, lo cual es eficiente computacionalmente.

### 📊 Fórmula:
$$
\min_{\beta_0, \beta_1} \sum_{i=1}^{n}(y_i - (\beta_0 + \beta_1 x_i))^2
$$

### 📌 Ejemplo gráfico:
Imagina que graficas puntos reales y la línea predicha. El método busca minimizar las distancias verticales al cuadrado entre los puntos reales y la línea.

---

## 📐 3. **Cálculo de la Pendiente (β₁)**

### 🔹 Definición:
La pendiente mide cuánto cambia Y por unidad de cambio en X. Representa la inclinación de la recta de regresión.

### ✅ Utilidad:
- Indica **la dirección y magnitud** de la relación entre variables.
- Es clave para interpretar el impacto de las variables independientes.

### 📊 Fórmula:
$$
\beta_1 = \frac{\sum{(x_i - \bar{x})(y_i - \bar{y})}}{\sum{(x_i - \bar{x})^2}} = \frac{\text{Cov}(X,Y)}{\text{Var}(X)}
$$

### 📌 Ejemplo:
Con los datos anteriores:

- $$ \bar{x} = 6 $$ $$ \bar{y} = 70 $$
- Calculamos covarianza y varianza manualmente:

  $$
  \text{Cov}(X,Y) = \frac{(2-6)(50-70) + (4-6)(60-70) + ...}{n-1}
  $$

  $$
  \text{Var}(X) = \frac{(2-6)^2 + (4-6)^2 + ...}{n-1}
  $$

  Finalmente:
  $$
  \beta_1 = \frac{\text{Cov}(X,Y)}{\text{Var}(X)} = 5
  $$

---

## ⚖️ 4. **Cálculo del Intercepto (β₀)**

### 🔹 Definición:
El intercepto representa el valor de Y cuando X = 0. Es el punto donde la recta cruza el eje Y.

### ✅ Utilidad:
- Es necesario para hacer predicciones incluso cuando no hay observaciones exactas en X=0.
- Completa el modelo de regresión junto con la pendiente.

### 📊 Fórmula:
$$
\beta_0 = \bar{y} - \beta_1 \cdot \bar{x}
$$

### 📌 Ejemplo:
Con $$ \bar{y} = 70 $$, $$ \bar{x} = 6 $$, $$ \beta_1 = 5 $$:

$$
\beta_0 = 70 - 5 \cdot 6 = 40
$$

---

## 🐍 5. **Ejemplo en Python con scikit-learn**

### 🔹 Definición:
Scikit-learn es una biblioteca de aprendizaje automático que permite implementar regresión lineal fácilmente.

### ✅ Utilidad:
- Rápida implementación de modelos predictivos.
- Fácil integración con otras herramientas de ML.
- Herramientas para métricas, validación cruzada, etc.

### 🧠 Código paso a paso:

```PYTHON
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Datos
X = [[1], [2], [3], [4], [5]]
y = [2, 4, 5, 4, 5]

# Modelo
model = LinearRegression()
model.fit(X, y)

# Coeficientes
print("Pendiente:", model.coef_[0])
print("Intercepto:", model.intercept_)

# Predicciones
y_pred = model.predict(X)
print("Predicciones:", y_pred)

# Métricas
print("MSE:", mean_squared_error(y, y_pred))
print("R²:", r2_score(y, y_pred))

```

### 📌 Aplicación real:
Este tipo de código se usa en sistemas de recomendación, precios dinámicos, análisis de tendencias, etc.

---

## 📊 6. **Métricas de error y evaluación**

### 🔹 Definición:
Son indicadores que miden qué tan bien funciona el modelo en comparación con los valores reales.

### ✅ Utilidad:
- Comparar modelos entre sí.
- Validar si están sobreajustando o subajustando.
- Comunicar resultados a equipos técnicos o no técnicos.

| Métrica                                            | Fórmula                                                             | Descripción                      |
| -------------------------------------------------- | ------------------------------------------------------------------- | -------------------------------- |
| **Error Cuadratico Medio (MSE)**                   | $$ \frac{1}{n} \sum{(y_i - \hat{y}_i)^2} $$                         | Promedio de errores al cuadrado  |
| **Errr absoluto medio (MAE)**                      | $$ \sqrt{\text{MSE}} $$                                             | Raíz cuadrada del MSE            |
| **Coeficiente de determinacion - R² (R-cuadrado)** | $$ 1 - \frac{\sum{(y_i - \hat{y}_i)^2}}{\sum{(y_i - \bar{y})^2}} $$ | Proporción de varianza explicada |

### 📌 Ejemplo:
Si R² = 0.95, significa que el modelo explica el 95% de la variación en Y.

---

## 🐍 7. **Ejemplo con Statsmodels**

### 🔹 Definición:
Statsmodels es una biblioteca enfocada en análisis estadístico, ideal para modelos inferenciales.

### ✅ Utilidad:
- Proporciona p-valores, intervalos de confianza, estadísticas de bondad de ajuste.
- Útil para investigación y análisis causal.

### 🧠 Código:

```PYTHON
import statsmodels.api as sm

# Datos
X = sm.add_constant([[1], [2], [3], [4], [5]])
y = [2, 4, 5, 4, 5]

# Modelo
model = sm.OLS(y, X).fit()
print(model.summary())

```

En el output:
- Puedes ver si una variable es significativa (p-valor < 0.05).
- Verificar si el modelo es útil usando el F-statistic.
- Revisar si hay colinealidad o heterocedasticidad.

---

## 📈 8. **Regresión Lineal Múltiple**

### 🔹 Definición:
Extensión de la regresión lineal simple, donde se usan múltiples variables predictoras.

### ✅ Utilidad:
- Modelar relaciones más complejas.
- Predecir resultados considerando varias causas.
- Mejor precisión que con solo una variable.

### 📊 Ecuación generalizada:
$$
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_n X_n + \epsilon
$$

### 🧠 Código:

```PYTHON
from sklearn.linear_model import LinearRegression

X_multi = [[1, 2], [2, 3], [3, 4], [4, 5]]
y_multi = [2, 4, 5, 4]

model = LinearRegression().fit(X_multi, y_multi)
print("Coeficientes:", model.coef_)  # [0.5, 0.5]
print("Intercepto:", model.intercept_)

```

Modelo final:
$$
Y = 0.5 X_1 + 0.5 X_2 + 0.0
$$

---

## ⚙️ 9. **Métodos de Selección del Modelo**

### 🔹 Definición:
Técnicas para seleccionar las variables más relevantes en una regresión lineal múltiple.

### ✅ Utilidad:
- Simplificar modelos.
- Evitar overfitting.
- Mejorar interpretabilidad.
- Reducir costos computacionales.

### 📌 Tipos:

#### ➤ Forward Selection
- Empieza sin variables.
- Agrega la que da mejor mejora (por AIC/BIC o p-valor).

#### ➤ Backward Elimination
- Empieza con todas las variables.
- Elimina la menos significativa (alta p-valor).

#### ➤ Stepwise Regression
- Mezcla de ambos: agrega y elimina variables iterativamente.

### 🧠 Código de Backward Elimination:

```PYTHON
def backward_elimination(X, y, sl=0.05):
    X = np.append(arr=np.ones((len(X), 1)), values=X, axis=1)
    for i in range(X.shape[1]):
        model = sm.OLS(y, X).fit()
        max_pval = max(model.pvalues)
        if max_pval > sl:
            j = np.argmax(model.pvalues)
            X = np.delete(X, j, axis=1)
        else:
            break
    return model, X

```

---

## 🎯 Resumen Final

| Concepto                      | Utilidad principal                                                                 |
|------------------------------|-------------------------------------------------------------------------------------|
| Regresión Lineal             | Modelar relación entre variables y hacer predicciones                               |
| Mínimos Cuadrados            | Encontrar los mejores coeficientes                                                  |
| Pendiente e Intercepto       | Interpretar cómo afectan las variables                                              |
| Scikit-learn                 | Implementar rápido y fácil                                                         |
| Métricas                     | Evaluar desempeño del modelo                                                      |
| Statsmodels                  | Obtener análisis estadístico completo                                               |
| Regresión Múltiple           | Trabajar con múltiples variables explicativas                                       |
| Métodos de selección         | Seleccionar variables relevantes para mejorar el modelo                             |
