import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage

try:
    df = pd.read_csv('dataset_generos_musicales.csv')
    print(df.head())
except FileNotFoundError:
    print("Error: El archivo 'dataset_generos_musicales.csv' no fue encontrado.")
    exit()


print(df.info())
print(df.describe())
df_features = df.set_index('País')

plt.figure(figsize=(10, 7))
sns.heatmap(df_features, annot=True, cmap='viridis', fmt='g')
plt.title('Popularidad de Géneros Musicales por País')
plt.xlabel('Géneros Musicales')
plt.ylabel('Países')
plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)


print("K-Means")

kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters_k3 = kmeans_3.fit_predict(X_scaled)
df['Cluster_K3'] = clusters_k3
print(f"Clusters asignados con K=3:\n{df[['País', 'Cluster_K3']]}\n")

inertia = []
k_range = range(1, 8)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(k_range, inertia, marker='o', linestyle='--')
plt.title('Método del Codo para determinar K óptimo')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Inercia')
plt.xticks(k_range)
plt.grid(True)
plt.show()

silhouette_scores = []
k_range_sil = range(2, 8)
for k in k_range_sil:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"Para K={k}, el coeficiente de silueta promedio es: {silhouette_avg:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(k_range_sil, silhouette_scores, marker='o', linestyle='--')
plt.title('Coeficiente de Silueta para determinar K óptimo')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Coeficiente de Silueta Promedio')
plt.xticks(k_range_sil)
plt.grid(True)
plt.show()

# Aplicaremos K-Means con K=2.
k_optimo = 2
kmeans_optimo = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
clusters_optimos = kmeans_optimo.fit_predict(X_scaled)
df['Cluster_K_Optimo'] = clusters_optimos
print(f"\nClusters asignados con K óptimo (K={k_optimo}):\n{df[['País', 'Cluster_K_Optimo']]}\n")

print("Clustering Jerárquico")

linked = linkage(X_scaled, method='ward')

plt.figure(figsize=(12, 7))
dendrogram(linked,
           orientation='top',
           labels=df_features.index,
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Dendrograma del Clustering Jerárquico')
plt.xlabel('Países')
plt.ylabel('Distancia Euclidiana (Ward)')
plt.show()


# Aplicar Clustering Jerárquico Aglomerativo con 2 clusters
agg_clustering = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')
clusters_agg = agg_clustering.fit_predict(X_scaled)
df['Cluster_Jerarquico'] = clusters_agg
print(f"Clusters asignados con Clustering Jerárquico (n=2):\n{df[['País', 'Cluster_Jerarquico']]}\n")

print("DBSCAN")

params = [(1.5, 2), (2.0, 2), (2.5, 3)]
for eps, min_samples in params:
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters_dbscan = dbscan.fit_predict(X_scaled)
    df[f'Cluster_DBSCAN_eps{eps}_ms{min_samples}'] = clusters_dbscan
    print(f"Resultados de DBSCAN con eps={eps} y min_samples={min_samples}:")
    print(f"Clusters: {clusters_dbscan}")
    print(f"Número de clusters encontrados: {len(np.unique(clusters_dbscan)) - (1 if -1 in clusters_dbscan else 0)}")
    print(f"Puntos de ruido (etiqueta -1): {np.sum(clusters_dbscan == -1)}\n")

print("3.1. PCA")

pca = PCA(n_components=None)
pca.fit(X_scaled)
varianza_explicada_acumulada = np.cumsum(pca.explained_variance_ratio_)

n_componentes_90 = np.argmax(varianza_explicada_acumulada >= 0.90) + 1
print(f"Varianza explicada acumulada por componente: {varianza_explicada_acumulada}")
print(f"Número de componentes para explicar al menos el 90% de la varianza: {n_componentes_90}\n")

pca_2d = PCA(n_components=2)
X_pca = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Cluster_K_Optimo'], s=200, palette='viridis')

for i, country in enumerate(df_features.index):
    plt.text(X_pca[i, 0] + 0.05, X_pca[i, 1] + 0.05, country, fontsize=12)

plt.title('Visualización de Países con PCA (2 Componentes Principales)')
plt.xlabel(f'Componente Principal 1 ({pca_2d.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'Componente Principal 2 ({pca_2d.explained_variance_ratio_[1]*100:.2f}%)')
plt.grid(True)
plt.axhline(0, color='grey', lw=0.5)
plt.axvline(0, color='grey', lw=0.5)
plt.show()


print("t-SNE")

perplexities = [2, 3]

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('Visualización de Países con t-SNE y diferente Perplexity', fontsize=16)

for i, perplexity in enumerate(perplexities):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_scaled)

    ax = axes[i]
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=df['Cluster_K_Optimo'], s=200, palette='viridis', ax=ax)

    for j, country in enumerate(df_features.index):
        ax.text(X_tsne[j, 0] + 0.1, X_tsne[j, 1] + 0.1, country, fontsize=12)

    ax.set_title(f't-SNE con Perplexity = {perplexity}')
    ax.set_xlabel('Dimensión 1')
    ax.set_ylabel('Dimensión 2')
    ax.grid(True)

plt.show()



print("""
--- 4.1. Comparación de Métodos ---

DIFERENCIAS ENTRE LOS ALGORITMOS DE CLUSTERIZACIÓN:

* **K-Means**: Es un algoritmo partitivo. Requiere que se especifique el número de clusters (K)
    de antemano. Funciona bien con clusters de forma esférica y de tamaño similar. Su objetivo
    es minimizar la suma de las distancias al cuadrado entre los puntos y el centroide de su cluster asignado (inercia).

* **Clustering Jerárquico**: Crea una jerarquía de clusters que puede ser representada como
    un dendrograma. No requiere especificar el número de clusters de antemano; se puede
    decidir después de ver el dendrograma. Hay dos enfoques: aglomerativo (empieza con cada
    punto como un cluster) y divisivo (empieza con un cluster y lo divide).

* **DBSCAN**: Es un algoritmo basado en densidad. Agrupa puntos que están muy juntos,
    marcando como ruido los puntos que se encuentran solos en regiones de baja densidad. No
    requiere especificar el número de clusters, sino dos parámetros: `eps` (distancia) y
    `min_samples` (densidad mínima). Puede encontrar clusters de formas arbitrarias.

¿CUÁL FUNCIONÓ MEJOR Y POR QUÉ?

En este caso, **K-Means y el Clustering Jerárquico funcionaron igual de bien**, ya que ambos
identificaron de manera consistente la misma estructura de dos clusters. La elección entre ellos
podría depender del objetivo:
- El **Clustering Jerárquico**, con su dendrograma, fue más útil para justificar visualmente
  la elección de dos clusters.
- **K-Means**, con el coeficiente de silueta, proporcionó una justificación numérica para
  la misma conclusión.

**DBSCAN fue el peor método** para este dataset. Debido al bajo número de muestras (8 países),
el concepto de "densidad" no tiene mucho sentido. El algoritmo tendía a clasificar
todos los puntos como ruido o como un único cluster, fallando en encontrar una
estructura significativa.
""")
print("""

COMPARACIÓN ENTRE PCA Y T-SNE:

* **PCA (Principal Component Analysis)**: Es una técnica lineal que busca proyectar los datos en
    un número menor de dimensiones (componentes principales) maximizando la varianza de los
    datos. Preserva las grandes distancias globales entre los puntos. Si dos puntos están lejos
    en el espacio original, probablemente estarán lejos en la proyección PCA.

* **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: Es una técnica no lineal que se enfoca
    en preservar las similitudes locales. Es excelente para visualizar la estructura de clusters,
    ya que intenta que los puntos cercanos en el espacio original también queden cerca en el
    mapa de baja dimensión. Sin embargo, las distancias globales en un gráfico t-SNE no son
    significativas (el tamaño de los clusters y la distancia entre ellos puede ser engañoso).

¿CUÁL VISUALIZÓ MEJOR LA RELACIÓN?

Para este dataset, **ambas técnicas fueron efectivas y mostraron resultados similares**,
visualizando claramente la separación en dos grupos.

- **PCA fue ligeramente mejor** en este caso porque, además de mostrar la separación, los ejes
  tienen una interpretación directa (combinaciones lineales de los géneros musicales que
  explican la mayor parte de la varianza). Dado que la estructura era simple, el poder de t-SNE
  para estructuras complejas no fue necesario.

- **t-SNE** confirmó la agrupación encontrada por PCA y los algoritmos de clustering, actuando
  como una validación visual sólida.
""")



# Calcular las medias de los géneros para cada cluster óptimo
cluster_analysis = df.groupby('Cluster_K_Optimo')[df_features.columns].mean().round(2)
print(cluster_analysis)
