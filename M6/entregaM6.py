import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, classification_report, roc_auc_score


try:
    df = pd.read_csv('cambio_climatico_agricultura.csv')
    print(df.head())
except FileNotFoundError:
    print("Error: El archivo 'cambio_climatico_agricultura.csv' no se encontró.")
    exit()

print(df.info())
print(df.describe())

# Se visualiza la distribución de cada variable numérica para detectar tendencias y valores atípicos.
print("\nVisualizando distribución de las variables...")
df.hist(bins=15, figsize=(15, 10), layout=(2, 2))
plt.suptitle('Distribución de las Variables Numéricas')
plt.show()

# Se utiliza un pairplot para visualizar las relaciones entre las variables y sus distribuciones.
# Esto ayuda a identificar correlaciones visuales entre las características climáticas y la producción.
print("\nVisualizando relaciones entre variables...")
sns.pairplot(df)
plt.suptitle('Relaciones Bivariadas entre Variables', y=1.02)
plt.show()

# =================================================================================
# 2. Preprocesamiento y escalamiento de datos (2 puntos)
# =================================================================================
print("\n2. Preprocesamiento y escalamiento de datos")

# Se definen las variables predictoras (X) y la variable objetivo (y) para la regresión.
# El país es un identificador, no una variable predictora en este contexto.
features = ['Temperatura_promedio', 'Cambio_lluvias', 'Frecuencia_sequías']
target_regression = 'Producción_alimentos'

X = df[features]
y_reg = df[target_regression]

# Se aplica estandarización a las variables numéricas (features).
# Esto es crucial para modelos sensibles a la escala como SVM y también es una buena práctica para la regresión.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Se dividen los datos en conjuntos de entrenamiento (80%) y prueba (20%).
X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X_scaled, y_reg, test_size=0.2, random_state=42
)

print(f"Datos divididos en {len(X_train)} registros de entrenamiento y {len(X_test)} de prueba.")


print("\n--- INICIO DE MODELOS DE REGRESIÓN ---")

regression_models = {
    "Regresión Lineal": LinearRegression(),
    "Árbol de Decisión": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42)
}


for name, model in regression_models.items():
    model.fit(X_train, y_reg_train)
    y_reg_pred = model.predict(X_test)

    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    mse = mean_squared_error(y_reg_test, y_reg_pred)
    r2 = r2_score(y_reg_test, y_reg_pred)

    print(f"\nResultados para el modelo: {name}")
    print(f"  Error Absoluto Medio (MAE): {mae:.2f}")
    print(f"  Error Cuadrático Medio (MSE): {mse:.2f}")
    print(f"  Coeficiente de Determinación (R²): {r2:.4f}")

print("\n--- INICIO DE MODELOS DE CLASIFICACIÓN ---")

# Se crea la variable categórica "Impacto_Climatico" basada en los cuantiles de la producción.
# Se asume que una menor producción implica un mayor impacto climático.
# 'Bajo' impacto = Alta producción
# 'Medio' impacto = Producción media
# 'Alto' impacto = Baja producción
df['Impacto_Climatico'] = pd.qcut(
    df['Producción_alimentos'],
    q=3,
    labels=['Alto', 'Medio', 'Bajo']
)
print("\nNueva variable 'Impacto_Climatico' creada:")
print(df[['País', 'Producción_alimentos', 'Impacto_Climatico']].head())

# Se define la nueva variable objetivo para la clasificación.
y_class_labels = df['Impacto_Climatico']

# Se codifican las etiquetas categóricas a valores numéricos.
label_encoder = LabelEncoder()
y_class = label_encoder.fit_transform(y_class_labels)

X_train_c, X_test_c, y_class_train, y_class_test = train_test_split(
    X_scaled, y_class, test_size=0.2, random_state=42, stratify=y_class
)

classification_models = {
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Árbol de Decisión": DecisionTreeClassifier(random_state=42),
    "Support Vector Machine": SVC(random_state=42, probability=True)
}

for name, model in classification_models.items():

    model.fit(X_train_c, y_class_train)

    y_class_pred = model.predict(X_test_c)
    y_class_proba = model.predict_proba(X_test_c) # Para ROC-AUC

    conf_matrix = confusion_matrix(y_class_test, y_class_pred)
    class_report = classification_report(y_class_test, y_class_pred, target_names=label_encoder.classes_)
    roc_auc = roc_auc_score(y_class_test, y_class_proba, multi_class='ovr')

    print(f"\nResultados para el modelo: {name}")
    print("  Matriz de Confusión:")
    print(conf_matrix)
    print("\n  Reporte de Clasificación:")
    print(class_report)
    print(f"  Curva ROC-AUC (One-vs-Rest): {roc_auc:.4f}")

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

rf_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf_reg, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)

grid_search.fit(X_train, y_reg_train)

print(f"Mejores parámetros encontrados: {grid_search.best_params_}")
print(f"Mejor puntaje R² (validación cruzada): {grid_search.best_score_:.4f}")

# Se evalúa el modelo optimizado en el conjunto de prueba.
best_rf_model = grid_search.best_estimator_
y_reg_pred_optimized = best_rf_model.predict(X_test)
r2_optimized = r2_score(y_reg_test, y_reg_pred_optimized)

print(f"R² en el conjunto de prueba con el modelo optimizado: {r2_optimized:.4f}")

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_reg_train)

linear_model = regression_models["Regresión Lineal"]
print(f"Coeficientes del modelo de Regresión Lineal: {linear_model.coef_}")
print(f"Coeficientes del modelo de Regresión Ridge (alpha=1.0): {ridge_model.coef_}")

# Se evalúa el impacto en el rendimiento.
y_pred_ridge = ridge_model.predict(X_test)
r2_ridge = r2_score(y_reg_test, y_pred_ridge)
print(f"R² del modelo Ridge en prueba: {r2_ridge:.4f}")
print("La regularización Ridge penaliza los coeficientes grandes para prevenir el sobreajuste,")
print("lo que puede mejorar la generalización del modelo, especialmente con datos complejos o colineales.")
