import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Sample dataset
X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11]])

# Step 1: Create dendrogram
linked = linkage(X, method='ward')

plt.figure(figsize=(6, 4))
dendrogram(linked)
plt.title("Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

# Step 2: Apply Agglomerative Clustering
hc = AgglomerativeClustering(n_clusters=2, linkage='ward')
labels = hc.fit_predict(X)

# Step 3: Plot clusters
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.title("Hierarchical Clustering")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()