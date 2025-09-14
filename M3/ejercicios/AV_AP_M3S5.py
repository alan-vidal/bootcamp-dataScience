import pandas as pd

datos = {
    'estudiante': ['Juan', 'Juan', 'Maria', 'Maria'],
    'materia': ['Matematicas', 'Historia', 'Matematicas', 'Historia'],
    'calificacion': [6.5, 5.8, 4.2, 6.0],
}

df = pd.DataFrame(datos)

df.columns = df.columns.str.strip() #Remueve espacios en blanco en los encabezados
df.columns = df.columns.str.upper()
print(f"🏁 DATASET ORIGINAL\n{df}")

df_new= df.set_index(['ESTUDIANTE', 'MATERIA'])
print(df_new)
print(df_new.loc['Maria', 'Historia'])

df_grouped= df.groupby('MATERIA')['CALIFICACION'].agg(["mean", "max"])
print(df_grouped)

df_pivot = df.pivot_table(index='ESTUDIANTE', columns='MATERIA', values='CALIFICACION')
print(df_pivot)

df_melted = df.melt(id_vars=['ESTUDIANTE', 'MATERIA'], value_vars=['CALIFICACION'], var_name='CALIFICACION', value_name='V_CALIFICACION')
print(df_melted)

print('---------------------')

d1 = {
    'ID_Estudiante': [1,2,3],
    'Estudiante': ['Juan', 'Maria', 'Pedro'],
    'Carrera': ['Ing. 1', 'Ing. 2', 'Ing. 3'],
}

d2 = {
    'ID_Estudiante': [1,2,3],
    'Materia': ['Matematicas', 'Matematicas', 'Matematicas'],
    'Calificación': [6.5, 5.8, 4.2],
}

df1 = pd.DataFrame(d1)
df2 = pd.DataFrame(d2)
print(df1)
print(df2)
df_concat = pd.concat([df1, df2], axis=0)
print(df_concat)
df_concat1 = pd.concat([df1, df2], axis=1)
print(df_concat1)

df_merge = pd.merge(df1, df2, on='ID_Estudiante')
print(df_merge)
