import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql.functions import col, desc, when, mean, stddev
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import os

print("Iniciando sesión de Spark...")
spark = SparkSession.builder \
    .appName("EvaluacionFinal_Migraciones") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("\n" + "="*60)
print("1. CARGA Y EXPLORACIÓN DE DATOS")
print("="*60)

schema = StructType([
    StructField("ID", IntegerType(), True),
    StructField("Origen", StringType(), True),
    StructField("Destino", StringType(), True),
    StructField("Año", IntegerType(), True),
    StructField("Razón", StringType(), True),
    StructField("PIB_Origen", DoubleType(), True),
    StructField("PIB_Destino", DoubleType(), True),
    StructField("Tasa_Desempleo_Origen", DoubleType(), True),
    StructField("Tasa_Desempleo_Destino", DoubleType(), True),
    StructField("Nivel_Educativo_Origen", DoubleType(), True),
    StructField("Nivel_Educativo_Destino", DoubleType(), True),
    StructField("Población_Origen", IntegerType(), True),
    StructField("Población_Destino", IntegerType(), True)
])

data_path = "migraciones.csv"
df_raw = spark.read.schema(schema).option("header", "true").csv(data_path, inferSchema=False)

df_raw.show(5, truncate=False)
df_raw.printSchema()
df_raw.describe().show()

rdd_raw = df_raw.rdd

for row in rdd_raw.take(3):
    print(row)

print("Transformaciones sobre RDDs:")

# Ejemplo 1: filter - Filtrar migraciones donde el PIB del destino es mayor a 40,000
rdd_filtered = rdd_raw.filter(lambda row: row.PIB_Destino > 40000)
print(f"Número de migraciones con PIB destino > 40,000: {rdd_filtered.count()}")

# Ejemplo 2: map - Crear una nueva columna calculada: Diferencia de PIB
rdd_pib_diff = rdd_raw.map(lambda row: (row.Origen, row.Destino, row.Año, row.PIB_Destino - row.PIB_Origen))
print("\nPrimeras 3 migraciones con diferencia de PIB:")
for item in rdd_pib_diff.take(3):
    print(f"{item[0]} -> {item[1]} ({item[2]}): ${item[3]:,.0f}")

# Ejemplo 3: flatMap - Separar razones en categorías más amplias
rdd_flat = rdd_raw.flatMap(lambda row: [(row.Origen, row.Razón), (row.Destino, row.Razón)])
print("\nPrimeras 5 pares (país, razón) usando flatMap:")
for pair in rdd_flat.take(5):
    print(pair)

# Acciones sobre RDDs
print("\nAcciones sobre RDDs:")
print(f"Total de registros en RDD: {rdd_raw.count()}")
print(f"Primeros 2 registros con collect: {rdd_raw.take(2)}")

# Operaciones con DataFrames: filtrado, agregaciones y ordenamiento
print("\nOperaciones con DataFrames:")

# Filtrado: migraciones por conflicto
df_conflict = df_raw.filter(col("Razón") == "Conflicto")
print("\nMigraciones por conflicto:")
df_conflict.show(truncate=False)

# Agregaciones: promedio de PIB por país de origen
df_avg_pib_origen = df_raw.groupBy("Origen").agg(
    mean("PIB_Origen").alias("Promedio_PIB_Origen"),
    mean("Tasa_Desempleo_Origen").alias("Promedio_Tasa_Desempleo_Origen"),
    mean("Nivel_Educativo_Origen").alias("Promedio_Nivel_Educativo_Origen"),
    sum("Población_Origen").alias("Total_Población_Origen")
).orderBy(desc("Promedio_PIB_Origen"))

print("\nPromedio de PIB, desempleo y nivel educativo por país de origen (ordenado por PIB):")
df_avg_pib_origen.show(truncate=False)

# Ordenamiento: top 5 países destino por PIB
df_top_destinos = df_raw.orderBy(desc("PIB_Destino")).select("Destino", "PIB_Destino").limit(5)
print("\nTop 5 países destino por PIB per cápita:")
df_top_destinos.show(truncate=False)

# Escribir resultados en formato Parquet
print("\nEscribiendo resultados procesados en formato Parquet...")
output_dir = "output_results"
os.makedirs(output_dir, exist_ok=True)

# Guardar el DataFrame filtrado de conflictos
df_conflict.write.mode("overwrite").parquet(f"{output_dir}/migraciones_conflicto.parquet")
print(f"Datos de migraciones por conflicto guardados en: {output_dir}/migraciones_conflicto.parquet")

# Guardar el resumen por país de origen
df_avg_pib_origen.write.mode("overwrite").parquet(f"{output_dir}/resumen_paises_origen.parquet")
print(f"Resumen por países de origen guardado en: {output_dir}/resumen_paises_origen.parquet")


# Registrar el DataFrame como tabla temporal
df_raw.createOrReplaceTempView("migraciones")

# Consulta 1: Países de origen y destino más frecuentes
print("\nConsulta 1: Top 3 países de origen y destino:")
spark.sql("""
    SELECT
        Origen,
        COUNT(*) as conteo_origen
    FROM migraciones
    GROUP BY Origen
    ORDER BY conteo_origen DESC
    LIMIT 3
""").show(truncate=False)

spark.sql("""
    SELECT
        Destino,
        COUNT(*) as conteo_destino
    FROM migraciones
    GROUP BY Destino
    ORDER BY conteo_destino DESC
    LIMIT 3
""").show(truncate=False)

# Consulta 2: Análisis de las principales razones de migración por región
print("\nConsulta 2: Principales razones de migración por país de origen:")
spark.sql("""
    SELECT
        Origen,
        Razón,
        COUNT(*) as cantidad
    FROM migraciones
    GROUP BY Origen, Razón
    ORDER BY Origen, cantidad DESC
""").show(truncate=False)

print("\nConsulta 2b: Razón más común por región (agrupado por origen)")
spark.sql("""
    WITH ranked_reasons AS (
        SELECT
            Origen,
            Razón,
            COUNT(*) as cantidad,
            ROW_NUMBER() OVER (PARTITION BY Origen ORDER BY COUNT(*) DESC) as rn
        FROM migraciones
        GROUP BY Origen, Razón
    )
    SELECT Origen, Razón, cantidad
    FROM ranked_reasons
    WHERE rn = 1
    ORDER BY cantidad DESC
""").show(truncate=False)

# Convertir la variable categórica 'Razón' en binaria: 1 si es 'Económica' o 'Laboral', 0 si es 'Conflicto' o 'Política'
# Objetivo: Predecir si la migración es por motivos económicos/laborales (1) vs otros (0)
df_ml = df_raw.withColumn(
    "label",
    when((col("Razón") == "Económica") | (col("Razón") == "Laboral"), 1.0)
    .otherwise(0.0)
)

# Seleccionar características numéricas para el modelo
feature_columns = [
    "PIB_Origen", "PIB_Destino",
    "Tasa_Desempleo_Origen", "Tasa_Desempleo_Destino",
    "Nivel_Educativo_Origen", "Nivel_Educativo_Destino",
    "Población_Origen", "Población_Destino"
]

# Crear un vector assembler para combinar todas las características en una sola columna
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df_assembled = assembler.transform(df_ml).select("label", "features")

# Verificar el nuevo DataFrame
print("\nEstructura del DataFrame listo para MLlib:")
df_assembled.printSchema()
df_assembled.show(5, truncate=False)

# Dividir los datos en entrenamiento y prueba (70% - 30%)
train_data, test_data = df_assembled.randomSplit([0.7, 0.3], seed=42)

print(f"\nNúmero de muestras de entrenamiento: {train_data.count()}")
print(f"Número de muestras de prueba: {test_data.count()}")

# Crear y entrenar el modelo de regresión logística
print("\nEntrenando modelo de Regresión Logística...")
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train_data)

# Hacer predicciones en el conjunto de prueba
predictions = lr_model.transform(test_data)

print("\nPredicciones del modelo (primeras 5 filas):")
predictions.select("label", "prediction", "probability").show(5, truncate=False)

# Evaluador de clasificación binaria (AUC)
evaluator_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc = evaluator_auc.evaluate(predictions)
print(f"Área bajo la curva ROC (AUC): {auc:.4f}")

# Evaluador de precisión general
evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator_acc.evaluate(predictions)
print(f"Precisión (Accuracy): {accuracy:.4f}")

# Precisión por clase
print("\nMatriz de confusión (valores aproximados):")
predictions.groupBy("label", "prediction").count().orderBy("label", "prediction").show()

# Extraer e interpretar coeficientes del modelo
print("\nCoeficientes del modelo de regresión logística:")
coefficients = lr_model.coefficients.toArray()
feature_importance = list(zip(feature_columns, coefficients))
feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

print("Importancia relativa de las variables (por magnitud del coeficiente):")
for feature, coef in feature_importance:
    print(f"  {feature}: {coef:.4f}")

spark.stop()
