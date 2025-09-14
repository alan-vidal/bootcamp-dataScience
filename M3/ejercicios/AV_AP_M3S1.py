import numpy as np

# 2 Crear vector con 10 elementos
vector = np.arange(1, 11)
print(vector)
print("--*--")

# 3 Matriz 3x3
matriz = np.random.rand(3, 3)
print(matriz)
print("--*--")

# 4 Matriz identidad 3x3
identidad = np.eye(3)
print(identidad)
print("--*--")

# 5 reshape (P2)
print(vector.reshape((2,5)))
print("--*--")

# 6 Seleccionar los elementos mayores a 5 del vector original y mostrarlos
print(vector[vector>5])
print("--*--")
# 7 Realizar una operación matemática entre arreglos
# Crea dos arreglos de tamaño 5 con arange() y súmalos.
# Muestra el resultado.
arr1 = np.arange(5)
arr2 = np.arange(5)
print(f"arr1 {arr1}")
print(f"arr2 {arr2}")
print(f"arr1 + arr2 {arr1 + arr2}")
print("--*--")

print(np.sqrt(arr1))
