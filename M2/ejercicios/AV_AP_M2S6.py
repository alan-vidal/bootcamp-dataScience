class Libro:
    def __init__(self, _titulo, _autor, _isbn):
        self._titulo = _titulo
        self._autor = _autor
        self._isbn = _isbn
    def get_titulo(self):
        return self._titulo
    def get_autor(self):
        return self._autor
    def get_isbn(self):
        return self._isbn
    def descripcion(self):
        print(f"Libro: {self._titulo}, Autor: {self._autor}, ISBN: {self._isbn}")

class Biblioteca:
    def __init__(self):
        self._libros = []
    def agregar_libro(self, libro):
        self._libros.append(libro)
    def mostrar_libros(self):
        for libro in self._libros:
            libro.descripcion()

def main():
    #libro1 = Libro("El Principito", "Antoine de Saint-Exupéry", "978-84-8348-254-9")
    libro2 = Libro("Cien años de soledad", "Gabriel García Márquez", "978-84-8348-254-9")
    libro3 = Libro("Don Quijote de la Mancha", "Miguel de Cervantes", "978-84-8348-254-9")
    libro4 = Libro("El Quijote", "Miguel de Cervantes", "978-84-8348-254-9")
    libro5 = Libro("El Aleph", "Jorge Luis Borges", "978-84-8348-254-9")

    libros = [Libro("El Principito", "Antoine de Saint-Exupéry", "978-84-8348-254-9"), libro2, libro3, libro4, libro5]

    biblioteca = Biblioteca()
    for libro in libros:
        biblioteca.agregar_libro(libro)
    biblioteca.mostrar_libros()

if __name__ == "__main__":
    main()
