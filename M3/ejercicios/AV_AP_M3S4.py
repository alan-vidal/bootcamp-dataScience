import pandas as pd
#import pyarrow
import matplotlib.pyplot as plt

PATH = '/Users/alanvidal/Desktop/BootCamp-CD/AP/ventasS4.csv'
df = pd.read_csv(PATH,
#engine="pyarrow", dtype_backend="pyarrow"
)

df.columns = df.columns.str.strip() #Remueve espacios en blanco en los encabezados
df.columns = df.columns.str.upper() #Convierte a mayúsculas los encabezados

#print(f"\n{df.info()}")
#print(f"\n{df.describe()}")
#print(f"\n{df.head()}")
print(f"🏁 DATASET ORIGINAL\n{df}")

#Detectar y eliminar registros duplicados (2 puntos).
# SE QUE EN LA PAUTA ESTE PUNTO VA DESPUES, PERO NO REMOVER LOS DUPLICADOS
# AHORA PUEDE (EN ESTE CASO, SI) ALTERA LA MODA Y MEDIA
print(f"\n👇 DETECTAMOS REGISTROS DUPLICDAOS: \n{df.duplicated()}")
df=df.drop_duplicates()

#dentificar y manejar valores perdidos (2 puntos).
print(f"\n👇 VALORES 'PERDIDOS' POR CATEGORIA: \n{df.isnull().sum()}")
df['PRECIO'] = df['PRECIO'].fillna(df["PRECIO"].mean())
df['CATEGORÍA'] = df['CATEGORÍA'].fillna(df["CATEGORÍA"].mode()[0])
print(f"\n👇 MANEJO DE DATOS 'PERDIDOS' REALIZADO: \n{df.isnull().sum()}")

#Detectar y manejar outliers en la columna "Cantidad" (2 puntos).
item = 'CANTIDAD'
Q1 = df[item].quantile(0.25)
Q3 = df[item].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_filtered = df[(df[item] >= lower_bound) & (df[item] <= upper_bound)]
print(f"\n👇 Nuevo DATAFRAME:\n{df_filtered}")

#Reemplazar valores incorrectos y modificar la estructura del DataFrame (2 puntos).
plt.boxplot(df_filtered[item],  showmeans=True)
plt.title(f'Boxplot of {item}')
plt.ylabel(f'Value of {item}')
plt.grid(color = 'blue', linestyle = '--', linewidth = 0.5)
plt.show()
