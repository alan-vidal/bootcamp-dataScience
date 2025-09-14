import math
import statistics
import random

def calcular_area_rectangulo(largo,ancho):
    return largo * ancho

def calcular_circunferencia(radio):
    return 2 * math.pi * radio

def calcular_promedio_avanzado(*numeros):
    return statistics.mean(numeros)

def generar_numeros_aleatorios(cantidad, maximo):
    return sorted([random.randint(1, maximo) for _ in range(cantidad)])

def main():
    while True:
        try:
            print('-------------------------------')
            print('Bienvenido!')
            print('Que operacion desea realizar:')
            print('1. Calcular área de un rectángulo')
            print('2. Calcular circunferencia de un círculo')
            print('3. Calcular promedio avanzado')
            print('4. Generar números aleatorios')
            print('-------------------------------')
            op = int(input('Ingrese el número de la opción deseada: '))
            print('-------------------------------')
            while True:
                try:
                    match op:
                        case 1:
                            largo = float(input("Ingrese el largo del rectángulo: "))
                            ancho = float(input("Ingrese el ancho del rectángulo: "))
                            print(f"El área del rectángulo es: {calcular_area_rectangulo(largo, ancho)}")
                            break
                        case 2:
                            radio = float(input("Ingrese el radio del círculo: "))
                            print(f"La circunferencia del círculo es: {calcular_circunferencia(radio)}")
                            break
                        case 3:
                            numeros = input("Ingrese los números separados por coma: ").split(',')
                            numeros = [float(num) for num in numeros]
                            print(f"El promedio avanzado es: {calcular_promedio_avanzado(*numeros)}")
                            break
                        case 4:
                            cantidad = int(input("Ingrese la cantidad de números aleatorios a generar: "))
                            maximo = int(input("Ingrese el valor máximo para los números aleatorios: "))
                            numeros_aleatorios = generar_numeros_aleatorios(cantidad, maximo)
                            print(numeros_aleatorios)
                            break
                        case _:
                            print("Opción inválida")
                            break
                except ValueError:
                    pass
                except EOFError:
                    break
            print('-------------------------------')
        except ValueError:
            pass
        except EOFError:
            break

if __name__ == "__main__":
    main()
