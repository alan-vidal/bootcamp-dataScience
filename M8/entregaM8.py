import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv('dataset_natalidad.csv')

print(df.head())
print(df.info())
print(df.describe())

print(f"\nValores nulos en cada columna:\n{df.isnull().sum()}")

#df_clean = df.drop(columns=['País'])

correlation_matrix = df_clean.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .8})
plt.title('Matriz de Correlación entre Variables Socioeconómicas y Tasa de Natalidad')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(4, 2, figsize=(16, 14))
fig.suptitle('Distribución de Variables Predictoras y Objetivo', fontsize=16)

variables = ['PIB_per_capita', 'Acceso_Salud', 'Nivel_Educativo', 'Tasa_Empleo_Femenino',
             'Edad_Maternidad', 'Urbanizacion', 'Tasa_Natalidad']

for i, var in enumerate(variables):
    row = i // 2
    col = i % 2
    axes[row, col].hist(df_clean[var], bins=20, color='skyblue', edgecolor='black')
    axes[row, col].set_title(f'Distribución de {var}')
    axes[row, col].set_xlabel(var)
    axes[row, col].set_ylabel('Frecuencia')

if len(variables) % 2 != 0:
    fig.delaxes(axes[3, 1])

plt.tight_layout()
plt.show()

target_corr = correlation_matrix['Tasa_Natalidad'].abs().sort_values(ascending=False)
print(target_corr[1:])  # Excluir la correlación consigo misma (1.0)

# Gráficos de dispersión para las 5 variables más correlacionadas
top_5_vars = target_corr[1:6].index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Relación entre Tasa de Natalidad y las 5 Variables Más Correlacionadas', fontsize=16)

for i, var in enumerate(top_5_vars):
    row = i // 3
    col = i % 3
    axes[row, col].scatter(df_clean[var], df_clean['Tasa_Natalidad'], alpha=0.7, color='purple')
    axes[row, col].set_title(f'{var} vs Tasa de Natalidad')
    axes[row, col].set_xlabel(var)
    axes[row, col].set_ylabel('Tasa de Natalidad')
    z = np.polyfit(df_clean[var], df_clean['Tasa_Natalidad'], 1)
    p = np.poly1d(z)
    axes[row, col].plot(df_clean[var], p(df_clean[var]), "r--", alpha=0.8)

if len(top_5_vars) < 6:
    fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.show()

X = df_clean.drop('Tasa_Natalidad', axis=1)
y = df_clean['Tasa_Natalidad']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42)  # 0.1765 * 0.85 ≈ 0.15

print(f"Dimensiones de los conjuntos:")
print(f"Entrenamiento: {X_train.shape}")
print(f"Validación: {X_val.shape}")
print(f"Prueba: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("\nEscalado completado.")

# Diseño de la red neuronal
# Estructura requerida:
# - Capa de entrada: tantas neuronas como variables predictoras (7)
# - Mínimo 2 capas ocultas
# - Capa de salida: 1 neurona (regresión)

model = Sequential([

    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),

    Dense(32, activation='relu'),
    Dropout(0.3),

    Dense(16, activation='relu'),
    Dropout(0.2),

    Dense(1, activation='linear')
])

# Compilar el modelo
# Optimizador: Adam
# Función de pérdida: Mean Squared Error (MSE)
# Métrica: Mean Absolute Error (MAE)

model.compile(
    optimizer=Adam(learning_rate=0.001),  # Experimentamos con learning rate
    loss='mean_squared_error',
    metrics=['mae']
)

print("\nResumen del modelo:")
model.summary()

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

print("\nEntrenando el modelo...")
history = model.fit(
    X_train_scaled, y_train,
    epochs=200,
    batch_size=16,
    validation_data=(X_val_scaled, y_val),
    callbacks=[early_stopping],
    verbose=1
)


# Evaluar el modelo en el conjunto de prueba
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Resultados en el conjunto de prueba:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de Validación')
plt.title('Pérdida del Modelo durante el Entrenamiento')
plt.xlabel('Época')
plt.ylabel('MSE')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='MAE de Entrenamiento')
plt.plot(history.history['val_mae'], label='MAE de Validación')
plt.title('MAE del Modelo durante el Entrenamiento')
plt.xlabel('Época')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Tasa de Natalidad Real')
plt.ylabel('Tasa de Natalidad Predicha')
plt.title('Predicciones vs Valores Reales')
plt.grid(True)
plt.show()

# Obtener los pesos de la primera capa oculta
weights_layer1 = model.layers[0].get_weights()[0]  # Forma: (7, 64)
# Calcular la importancia promedio de cada variable a través de las conexiones a la primera capa oculta
feature_importance = np.mean(np.abs(weights_layer1), axis=1)
feature_names = X.columns

# Crear un DataFrame con la importancia
importance_df = pd.DataFrame({
    'Variable': feature_names,
    'Importancia': feature_importance
}).sort_values('Importancia', ascending=False)

print("\nTop 5 variables más influyentes (según peso promedio en la primera capa):")
print(importance_df.head())

# Visualización de la importancia de las variables
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df.head(8), x='Importancia', y='Variable', palette='viridis')
plt.title('Importancia Relativa de Variables en la Predicción de Natalidad')
plt.xlabel('Importancia Promedio (Absoluta)')
plt.ylabel('Variable')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

# Comparar predicciones con valores reales en algunos ejemplos
print("\n=== Ejemplos de Predicciones vs Valores Reales ===")
comparison_df = pd.DataFrame({
    'Real': y_test.values,
    'Predicho': y_pred.flatten(),
    'Error': np.abs(y_test.values - y_pred.flatten())
}).head(10)

print(comparison_df.round(3))

print("Variables más influyentes en la predicción de la tasa de natalidad:")
print(f"1. {importance_df.iloc[0]['Variable']}: {importance_df.iloc[0]['Importancia']:.4f}")
print(f"2. {importance_df.iloc[1]['Variable']}: {importance_df.iloc[1]['Importancia']:.4f}")
print(f"3. {importance_df.iloc[2]['Variable']}: {importance_df.iloc[2]['Importancia']:.4f}")
print(f"4. {importance_df.iloc[3]['Variable']}: {importance_df.iloc[3]['Importancia']:.4f}")
print(f"5. {importance_df.iloc[4]['Variable']}: {importance_df.iloc[4]['Importancia']:.4f}")
