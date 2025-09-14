import locale
import time
locale.setlocale(locale.LC_ALL, 'es_ES.UTF-8')

global biblioteca

class Libro:
    def __init__(self, _titulo, _autor, _precio, _cantidad):
        self._titulo = _titulo
        self._autor = _autor
        self._precio = _precio
        self._cantidad = _cantidad
    def get_titulo(self):
        return self._titulo
    def get_autor(self):
        return self._autor
    def get_precio(self):
        return self._precio
    def get_cantidad(self):
        return self._cantidad
    def descripcion(self):
        print(f"Titulo: {self._titulo}, Autor: {self._autor}, Precio: {locale.currency(self._precio)}, Cantidad: {self._cantidad}")

class Biblioteca:
    def __init__(self):
        self._libros = []
    def agregar_libro(self, libro):
        self._libros.append(libro)
    def mostrar_libros(self):
        for libro in self._libros:
            libro.descripcion()

#Carga datos para inicializar el sistema
def inicializar():
    libros = [
        Libro("El Principito", "Antoine de Saint-Exupéry", 15000, 10),
        Libro("Cien años de soledad", "Gabriel García Márquez", 20000, 5),
        Libro("Don Quijote de la Mancha", "Miguel de Cervantes", 18000, 8),
        Libro("El Quijote", "Miguel de Cervantes", 18000, 8),
        Libro("El Aleph", "Jorge Luis Borges", 25000, 3),
        Libro("Ficciones", "Jorge Luis Borges", 20000, 5)
    ]
    biblioteca = Biblioteca()
    for libro in libros:
        biblioteca.agregar_libro(libro)
    return biblioteca

def mostrar_libros_disponibles(biblioteca):
    print("\n---Libros disponibles:---")
    for libro in biblioteca._libros:
        if libro._cantidad > 0:
            libro.descripcion()

def menu_deley(txt):
    for i in range(4):
        print(f"Ingresando como {txt} {i*"."}", end='\r')
        time.sleep(0.8)

def menu():
    print("---¡Bienvenido!---")
    print("Desea ingresar como:")
    print("1. Usuario")
    print("2. Administrador\n")

    while True:
        try:
            option=input("Ingrese una opción: ")
            option = int(option)
            match option:
                case 1:
                    menu_deley("usuario")
                    break
                case 2:
                    menu_deley("administrador")
                    break
                case _:
                    print("Opción inválida")
                    pass
        except ValueError:
            print("Opción inválida")
            pass
        except EOFError:
            print("\n \n ---¡Adios!---\n")
            break

    return option

def main():
    #CARGA DE LIBROS
    biblioteca = inicializar()

    #MOSTRAR LIBROS
    #mostrar_libros_disponibles(biblioteca)

    menu_selection =menu() #Menu() retorna 1/2
    if menu_selection == 1:
        print("---Sistema de Compras---")
        print("1. Mostrar libros disponibles")
        print("2. Filtrar libro por rango de precios")
        print("3. Comprar libro")
        # COMPRAR LIBRO, ELEGUIR LIBRO POR TITULO, DEFINIR CANTIDAD - RESTAR CANTIDAD - MOSTRAR NUEVO STOCK - ERRORES - DESCUENTO
        # FUNCION FACTURA
        #
        #
    elif menu_selection == 2:
        print("ingresaste como administrador")
    else:
        print("ingresaste como invitado")
 # Use \n to move to a new line after the loop

if __name__ == "__main__":
    main()
