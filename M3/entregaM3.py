import pandas as pd
#import numpy as np

df = pd.read_csv('migracion.csv')

print(f"Dimensiones del DataFrame: {df.shape}")
print(df.head())

print(df.isnull().sum())

Q1 = df['Cantidad_Migrantes'].quantile(0.25)
Q3 = df['Cantidad_Migrantes'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Cantidad_Migrantes'] < lower_bound) | (df['Cantidad_Migrantes'] > upper_bound)]

print(f"\nLímite inferior: {lower_bound:.2f}")
print(f"Límite superior: {upper_bound:.2f}")
print(f"Número de outliers detectados en 'Cantidad_Migrantes': {len(outliers)}")

df = df[(df['Cantidad_Migrantes'] >= lower_bound) & (df['Cantidad_Migrantes'] <= upper_bound)]

mapeo_razon = {
    'Económica': 'Trabajo',
    'Conflicto': 'Guerra',
    'Educativa': 'Estudio'
}

df['Razon_Migracion'] = df['Razon_Migracion'].map(mapeo_razon)
print("\nValores únicos en 'Razon_Migracion' después del mapeo:")
print(df['Razon_Migracion'].unique())

#print(df.head())

print("\nInformación general del dataset:")
print(df.info())

print("\nEstadísticas descriptivas del dataset:")
print(df.describe())

media_migrantes = df['Cantidad_Migrantes'].mean()
mediana_migrantes = df['Cantidad_Migrantes'].median()

print(f"\nMedia de migrantes: {media_migrantes:,.0f}")
print(f"Mediana de migrantes: {mediana_migrantes:,.0f}")

pib_origen_promedio = df['PIB_Origen'].mean()
pib_destino_promedio = df['PIB_Destino'].mean()

print(f"PIB promedio de países de origen: ${pib_origen_promedio:,.0f}")
print(f"PIB promedio de países de destino: ${pib_destino_promedio:,.0f}")

conteo_razones = df['Razon_Migracion'].value_counts()
print(f"\nConteo de migraciones por razón:")
print(conteo_razones)

suma_migrantes_por_razon = df.groupby('Razon_Migracion')['Cantidad_Migrantes'].sum().reset_index()
print("Suma total de migrantes por razón de migración:")
print(suma_migrantes_por_razon)

media_idh_origen_por_razon = df.groupby('Razon_Migracion')['IDH_Origen'].mean().reset_index()
print("\nMedia del IDH de los países de origen por razón de migración:")
print(media_idh_origen_por_razon)

resumen_razon = suma_migrantes_por_razon.merge(media_idh_origen_por_razon, on='Razon_Migracion')
resumen_razon = resumen_razon.sort_values(by='Cantidad_Migrantes', ascending=False).reset_index(drop=True)
print("\nResumen ordenado de mayor a menor cantidad de migrantes:")
print(resumen_razon)

migraciones_guerra = df[df['Razon_Migracion'] == 'Guerra']
print("Migraciones por conflicto (Guerra):")
print(migraciones_guerra)

idh_destino_alto = df[df['IDH_Destino'] > 0.90]
print("\nMigraciones donde el IDH del país de destino es mayor a 0.90:")
print(idh_destino_alto)

df['Diferencia_IDH'] = df['IDH_Destino'] - df['IDH_Origen']
print("\nDataFrame con nueva columna 'Diferencia_IDH':")
print(df[['Pais_Origen', 'Pais_Destino', 'IDH_Origen', 'IDH_Destino', 'Diferencia_IDH']].head())

df.to_csv('migracion_limpio.csv', index=False)
print("Archivo 'migracion_limpio.csv' guardado exitosamente.")
