#Eres un trabajador de una tienda en línea que gestiona pedidos y el inventario de productos. Te
#solicitan realizar un programa en Python que permita gestionar y organizar los productos en
#diferentes estructuras de datos, con el fin de facilitar la búsqueda, organización y agrupación de la
#información.
from http.cookiejar import uppercase_escaped_char

def show_productos(productos):
    print("Productos disponibles:")
    for producto in productos:
        print(f"- {producto}")

def ingreso_inventario(productos):
    inventario = {}
    print("-------------------------")
    print("--Control de Inventario--")
    for producto in productos:
        cantidad = int(input(f"Ingrese la cantidad de {producto}: "))
        inventario[producto] = cantidad
    print("-------------------------")
    print("--Inventario Actualizado--")
    for producto, cantidad in inventario.items():
        print(f"{producto}: {cantidad}")

def main():
    productos = ["Super 8", "Choquita", "ObaOba","Ricolate", "Triton"]
    productos.sort()
    show_productos(productos)

    for i in range(2):
        productos.append(input("Ingrese un nuevo producto: "))
    productos.sort()
    show_productos(productos)
    ingreso_inventario(productos)

    #categorias = ("Golosinas", "Bebidas", "Snacks", "Otros")
    categorias ="Golosinas", "Bebidas", "Snacks", "Otros"
    c1 , c2 , c3 , c4 = categorias
    print("-------------------------")
    print("--Categorias--")
    print(c1)
    print(c2)
    print(c3)
    print(c4)

    print("-------------------------")
    print("--Productos (Lista Unica)--")
    productos_unicos = set(productos)
    print(productos_unicos)

    print("-------------------------")
    print("--Productos (Mayusculas)--")
    productos_mayusculas = [producto.upper() for producto in productos]
    print(productos_mayusculas)

if __name__ == "__main__":
    main()
