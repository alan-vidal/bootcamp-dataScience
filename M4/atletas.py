import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv('olimpicos.csv')
print(df.head())

print("\nInformación general del dataset:")
print(df.info())

print("\nEstadísticas descriptivas del dataset:")
print(df.describe())

# Genera un histograma del número de entrenamientos semanales
plt.figure(figsize=(10, 6))
plt.hist(df['Entrenamientos_Semanales'], bins=6, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Distribución de Entrenamientos Semanales', fontsize=16, fontweight='bold')
plt.xlabel('Número de Entrenamientos Semanales', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(range(3, 11))
plt.show()

print("Tipos de variables por columna:")
print(df.dtypes)


media_medallas = df['Medallas_Totales'].mean()
mediana_medallas = df['Medallas_Totales'].median()
moda_medallas = df['Medallas_Totales'].mode()[0]

print(f"\nMedia de medallas totales: {media_medallas:.2f}")
print(f"Mediana de medallas totales: {mediana_medallas}")
print(f"Moda de medallas totales: {moda_medallas}")

desv_est_altura = df['Altura_cm'].std()
print(f"\nDesviación estándar de la altura (cm): {desv_est_altura:.2f}")

plt.figure(figsize=(8, 6))
sns.boxplot(y=df['Peso_kg'], color='lightcoral')
plt.title('Boxplot de Peso de los Atletas (kg)', fontsize=16, fontweight='bold')
plt.ylabel('Peso (kg)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Calcular outliers usando IQR para Peso_kg
#Q1_peso = df['Peso_kg'].quantile(0.25)
#Q3_peso = df['Peso_kg'].quantile(0.75)
#IQR_peso = Q3_peso - Q1_peso
#lower_bound_peso = Q1_peso - 1.5 * IQR_peso
#upper_bound_peso = Q3_peso + 1.5 * IQR_peso

#outliers_peso = df[(df['Peso_kg'] < lower_bound_peso) | (df['Peso_kg'] > upper_bound_peso)]
#print(f"\nValores atípicos en Peso_kg (usando IQR): {len(outliers_peso)}")
#if len(outliers_peso) > 0:
#    print("Filas con peso atípico:")
#    print(outliers_peso[['Atleta', 'Peso_kg']])
#else:
#    print("No se encontraron valores atípicos en la columna Peso_kg.")

correlacion_pearson = df['Entrenamientos_Semanales'].corr(df['Medallas_Totales'])
print(f"Correlación de Pearson entre Entrenamientos Semanales y Medallas Totales: {correlacion_pearson:.4f}")

# Explicación de la correlación
if correlacion_pearson > 0.7:
    explicacion_corr = "Existe una fuerte correlación positiva."
elif correlacion_pearson > 0.3:
    explicacion_corr = "Existe una correlación positiva moderada."
elif correlacion_pearson > 0:
    explicacion_corr = "Existe una leve correlación positiva."
elif correlacion_pearson == 0:
    explicacion_corr = "No existe correlación lineal."
else:
    explicacion_corr = "Existe una correlación negativa."

print(f"Interpretación: {explicacion_corr}")

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Peso_kg', y='Medallas_Totales', color='purple', s=100, edgecolor='black')
plt.title('Relación entre Peso y Medallas Totales', fontsize=16, fontweight='bold')
plt.xlabel('Peso (kg)', fontsize=12)
plt.ylabel('Medallas Totales', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

X = df[['Entrenamientos_Semanales']]  # Variable independiente (característica)
y = df['Medallas_Totales']            # Variable dependiente (objetivo)

# Crear y entrenar el modelo
modelo_regresion = LinearRegression()
modelo_regresion.fit(X, y)

# Obtener los coeficientes de regresión
coeficiente = modelo_regresion.coef_[0]
intercepto = modelo_regresion.intercept_

print(f"Coeficiente del modelo: {coeficiente:.4f}")
print(f"Intercepto: {intercepto:.4f}")
print(f"Ecuación del modelo: Medallas_Totales = {coeficiente:.4f} * Entrenamientos_Semanales + {intercepto:.4f}")

# Interpreta el resultado
interpretacion_coef = f"Por cada aumento de 1 entrenamiento semanal, las medallas totales aumentan en promedio {coeficiente:.4f}."
print(f"Interpretación: {interpretacion_coef}")

# Calcula el R² para medir el ajuste del modelo
y_pred = modelo_regresion.predict(X)
r2 = r2_score(y, y_pred)
print(f"\nCoeficiente de determinación (R²): {r2:.4f}")

if r2 > 0.7:
    interpretacion_r2 = "El modelo explica una gran proporción de la variabilidad en las medallas."
elif r2 > 0.5:
    interpretacion_r2 = "El modelo explica una cantidad moderada de la variabilidad en las medallas."
else:
    interpretacion_r2 = "El modelo explica una baja proporción de la variabilidad en las medallas."
print(f"Interpretación de R²: {interpretacion_r2}")

# Usa Seaborn (regplot) para graficar la regresión lineal
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='Entrenamientos_Semanales', y='Medallas_Totales',
            scatter_kws={'color': 'blue', 's': 100},
            line_kws={'color': 'red', 'linewidth': 2})
plt.title('Regresión Lineal: Entrenamientos Semanales vs Medallas Totales', fontsize=16, fontweight='bold')
plt.xlabel('Entrenamientos Semanales', fontsize=12)
plt.ylabel('Medallas Totales', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

df_numericas = df.select_dtypes(include=[np.number])
correlation_matrix = df_numericas.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": 0.8})
plt.title('Heatmap de Correlación entre Variables Numéricas', fontsize=16, fontweight='bold')
plt.show()

# Crea un boxplot de la cantidad de medallas por disciplina deportiva
plt.figure(figsize=(12, 8))
sns.boxplot(data=df, x='Deporte', y='Medallas_Totales', palette='Set2')
plt.title('Distribución de Medallas Totales por Disciplina Deportiva', fontsize=16, fontweight='bold')
plt.xlabel('Disciplina Deportiva', fontsize=12)
plt.ylabel('Medallas Totales', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
