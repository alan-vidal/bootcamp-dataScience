import pandas as pd

datos = {
    'jugador': ['Lionel Messi', 'Cristiano Ronaldo', 'Kevin De Bruyne', 'Kylian Mbappe', 'Luka Modric'],
    'posicion': ['Delantero', 'Delantero', 'Mediocampista', 'Delantero', 'Mediocampista'],
    'edad': [35, 38, 31,24,37],
    'goles': [20, 18, 8,25,3],
    'asistencias': [10, 5, 15,12,8]
}

tabla = pd.DataFrame(datos)

# Selecciona una columna y muestra los nombres de todos los jugadores (1 punto).
print(tabla['jugador'])

# Filtra jugadores con más de 10 goles y muestra solo su nombre y cantidad de goles (1 punto).
print(f"\n Jugadores con mas de 10 goles:\n{tabla[tabla['goles'] > 10][['jugador', 'goles']]}")
# Agrega una nueva columna al DataFrame llamada Puntos, donde cada jugador obtiene Puntos = (Goles * 4) + (Asistencias * 2) (1 punto).
tabla['puntos'] = tabla['goles'] * 4 + tabla['asistencias'] * 2
print(f"\nTabla con Puntos: \n {tabla}")
# Calcula el promedio de goles de todos los jugadores (1 punto).
promedio_goles = tabla['goles'].mean()
print(f"\n Promedio de Goles: {promedio_goles}")
# Obtén el máximo y mínimo de asistencias en el equipo (1 punto).
max_asistencias = tabla['asistencias'].max()
min_asistencias = tabla['asistencias'].min()
print(f"\n Maximo de Asistencias: {max_asistencias}")
print(f"\n Minimo de Asistencias: {min_asistencias}")
# Cuenta cuántos jugadores hay por posición (Delantero, Mediocampista) (1 punto).
conteo_posiciones = tabla['posicion'].value_counts()
print(f"Conteo de Posiciones: {conteo_posiciones}")
# Ordena el DataFrame en función de los goles en orden descendente (1 punto).
tabla_ordenada = tabla.sort_values(by='goles', ascending=False)
print(f"\nTabla Ordenada por Goles: \n {tabla_ordenada}")
# Aplica describe() para obtener estadísticas generales del DataFrame (1 punto).
print(f"\nEstadísticas Generales:\n {tabla.describe()}")
# Usa value_counts() para contar cuántos jugadores hay en cada posición (1 punto).
conteo_posiciones = tabla['posicion'].value_counts()
print(f"\nConteo de Posiciones: {conteo_posiciones}")
