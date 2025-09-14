import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 1. Diseño del Experimento
#
# a) Asignación aleatoria: Asegurarse de que los 30 estudiantes sean asignados
#    completamente al azar al Grupo A o al Grupo B. Esto minimiza el sesgo de
#    selección, garantizando que las características preexistentes de los
#    estudiantes (como su habilidad innata o motivación) se distribuyan de
#    manera uniforme entre ambos grupos.
#
# b) Muestra representativa: La muestra de 30 estudiantes debería ser
#    representativa de la población total de estudiantes de la institución.
#    Si la muestra no es representativa, los resultados podrían no ser
#    generalizables a todos los estudiantes.
#
# c) Enmascaramiento (Ciego): Si es posible, los evaluadores que califican el
#    examen estándar no deberían saber a qué grupo pertenece cada estudiante.
#    Esto evita el sesgo del observador, donde las expectativas del evaluador
#    podrían influir en la calificación.
#
# d) Control de variables de confusión: Se deben identificar y controlar otras
#    variables que podrían afectar el rendimiento académico, como las horas de
#    estudio previas, el nivel socioeconómico o la asistencia a clases.
#    Esto se puede lograr mediante la recopilación de estos datos y su
#    inclusión en el análisis estadístico como covariables.

# ---------------------------------------------------------------------------

# Datos del problema
grupo_a_tutoria = np.array([85, 90, 78, 88, 92, 80, 86, 89, 84, 87, 91, 82, 83, 85, 88])
grupo_b_control = np.array([70, 72, 75, 78, 80, 68, 74, 76, 79, 77, 73, 71, 75, 78, 80])

# Media de cada grupo
media_grupo_a = np.mean(grupo_a_tutoria)
media_grupo_b = np.mean(grupo_b_control)

# Desviación estándar de cada grupo
std_grupo_a = np.std(grupo_a_tutoria, ddof=1)
std_grupo_b = np.std(grupo_b_control, ddof=1)

print("--- 2. Estadísticas Descriptivas ---")
print(f"Grupo A (Tutoría) - Media: {media_grupo_a:.2f}")
print(f"Grupo A (Tutoría) - Desviación Estándar: {std_grupo_a:.2f}")
print(f"Grupo B (Control) - Media: {media_grupo_b:.2f}")
print(f"Grupo B (Control) - Desviación Estándar: {std_grupo_b:.2f}")
print("-" * 35)

# Representación gráfica de los datos
plt.figure(figsize=(14, 6))

# Histograma para el Grupo A
plt.subplot(1, 2, 1)
plt.hist(grupo_a_tutoria, bins=5, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Histograma - Grupo A (Tutoría)')
plt.xlabel('Calificaciones')
plt.ylabel('Frecuencia')
plt.axvline(media_grupo_a, color='red', linestyle='dashed', linewidth=1, label=f'Media: {media_grupo_a:.2f}')
plt.legend()

# Histograma para el Grupo B
plt.subplot(1, 2, 2)
plt.hist(grupo_b_control, bins=5, color='salmon', edgecolor='black', alpha=0.7)
plt.title('Histograma - Grupo B (Control)')
plt.xlabel('Calificaciones')
plt.ylabel('Frecuencia')
plt.axvline(media_grupo_b, color='red', linestyle='dashed', linewidth=1, label=f'Media: {media_grupo_b:.2f}')
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.boxplot([grupo_a_tutoria, grupo_b_control], labels=['Grupo A (Tutoría)', 'Grupo B (Control)'])
plt.title('Diagrama de Caja Comparativo')
plt.ylabel('Calificaciones')
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.show()

# Se plantea la hipótesis nula y alternativa y se realiza una prueba t.

# Hipótesis Nula (H0): No hay diferencia en el rendimiento académico promedio
# entre el grupo de tutoría y el grupo de control. Matemáticamente: μ_A = μ_B
H0 = "No hay diferencia en el rendimiento académico entre los dos grupos."

# Hipótesis Alternativa (H1): El grupo con tutoría tiene un rendimiento
# académico promedio mejor (mayor) que el grupo de control.
# Matemáticamente: μ_A > μ_B
H1 = "El grupo con tutoría tiene un mejor rendimiento académico."

# Nivel de significancia
alpha = 0.05
t_statistic, p_value = stats.ttest_ind(grupo_a_tutoria, grupo_b_control, alternative='greater')

print("\n--- 3. Prueba de Hipótesis ---")
print(f"Hipótesis Nula (H0): {H0}")
print(f"Hipótesis Alternativa (H1): {H1}")
print(f"\nEstadístico t: {t_statistic:.4f}")
print(f"Valor-p: {p_value:.4f}")

if p_value < alpha:
    print(f"\nDecisión: Rechazar la hipótesis nula (p-valor < {alpha}).")
    print("Interpretación: Hay evidencia estadística suficiente para concluir que el grupo con tutoría tiene un rendimiento académico significativamente mejor que el grupo de control.")
else:
    print(f"\nDecisión: No rechazar la hipótesis nula (p-valor >= {alpha}).")
    print("Interpretación: No hay evidencia estadística suficiente para concluir que el grupo con tutoría tiene un rendimiento académico mejor que el grupo de control.")
print("-" * 35)

# Grados de libertad (gl) para la prueba t de Student con muestras independientes
n1, n2 = len(grupo_a_tutoria), len(grupo_b_control)
gl = n1 + n2 - 2

# Diferencia de las medias muestrales
diff_means = media_grupo_a - media_grupo_b

# Error estándar de la diferencia de medias
s1_sq = std_grupo_a**2
s2_sq = std_grupo_b**2
std_error_diff = np.sqrt((s1_sq / n1) + (s2_sq / n2))

# Valor crítico de t para un intervalo de confianza del 95%
# (1 - alpha) = 0.95
confidence_level = 0.95
alpha_ic = 1 - confidence_level
# Para un intervalo bilateral, se usa alpha/2.
t_critico = stats.t.ppf(1 - alpha_ic / 2, df=gl)

# Margen de error
margen_error = t_critico * std_error_diff

# Cálculo del intervalo de confianza
limite_inferior = diff_means - margen_error
limite_superior = diff_means + margen_error

print("\n--- 4. Intervalo de Confianza (95%) ---")
print(f"La diferencia de medias entre Grupo A y Grupo B es: {diff_means:.2f}")
print(f"El intervalo de confianza del 95% para la diferencia de medias es: ({limite_inferior:.2f}, {limite_superior:.2f})")

# Interpretación del intervalo de confianza
print("\nInterpretación del resultado:")
if limite_inferior > 0 and limite_superior > 0:
    print("Dado que todo el intervalo de confianza es positivo (no incluye el cero), podemos estar 95% seguros de que la media del rendimiento del Grupo A (Tutoría) es mayor que la del Grupo B (Control).")
    print(f"Específicamente, la mejora promedio se estima entre {limite_inferior:.2f} y {limite_superior:.2f} puntos.")
else:
    print("Dado que el intervalo de confianza incluye el cero, no podemos concluir con un 95% de confianza que exista una diferencia significativa entre las medias de los dos grupos.")
