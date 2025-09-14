import statistics

def estado(nota):
    """Calcula si estas aprobado o no"""
    if nota >= 60:
        return "Aprobado"
    else:
        return "Reprobado"

def comentario(nota):
    """Entrega un comentario en relacion a la nota que obtiene el estudiante"""
    if nota>= 90:
        return "Excelente"
    elif 75 <= nota < 89:
        return "Bueno"
    else:
        return "Necesita mejorar"

def main():

    materias = []
    estudiante = {}
    notas = []
    n = 0
    averege = []

    profe = input("Cual es tu nombre Profesor: ")
    evaluaciones = int(input("¿Cuantas Materias desea evaluar?: "))

    for i in range(evaluaciones):
        materias.append(str(input(f"{profe} ¿Cual es la {i+1}ra asignatura que evaluara?: ")))

    print("------------------")

    while True:
        try:
            est = str(input(f"{profe},ingrese Nombre Estudiante: "))
            estudiante = {"nombre":est}
            for materia in materias:
                nota = input(f"{profe},ingrese la nota que obtuvo {est} en {materia}: ")
                estudiante[f"{materia}"] = nota
            notas.append(estudiante)

            for materia in materias:
                averege.append(int(notas[n][f'{materia}']))

            print(f"{notas[n]['nombre']} obtuvo un PROMEDIO de {statistics.mean(averege)} - El estudiante esta {estado(statistics.mean(averege))}, {comentario(statistics.mean(averege))}")
            n+=1
        except ValueError:
            pass
        except EOFError:
            break

if __name__=="__main__":
    main()
