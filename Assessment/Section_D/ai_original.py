"""
Section D — STEP 1: AI original draft (intentionally uncorrected)

Typical issues left in this version:
- K-Means on raw unscaled features
- Elbow loop starts at k=1
- No random_state (non-reproducible)
- PCA scatter axes not labelled PC1 / PC2
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

rng = np.random.default_rng(0)
monthly_orders = rng.integers(1, 30, size=160)
avg_spend = rng.normal(350, 120, size=160).clip(50, 900)
avg_rating = rng.uniform(2.5, 5.0, size=160)
X = np.column_stack([monthly_orders, avg_spend, avg_rating])

inertias = []
ks = range(1, 11)
for k in ks:
    km = KMeans(n_clusters=k)
    km.fit(X)
    inertias.append(km.inertia_)

plt.figure()
plt.plot(list(ks), inertias, marker="o")
plt.title("Elbow Method")
plt.xlabel("k")
plt.ylabel("inertia")
plt.show()

best_k = 4
model = KMeans(n_clusters=best_k)
labels = model.fit_predict(X)

print("Cluster counts:")
for i in range(best_k):
    print(i, int((labels == i).sum()))

print("Centroids:")
print(model.cluster_centers_)

pca = PCA(n_components=2)
coords = pca.fit_transform(X)
plt.figure()
for i in range(best_k):
    plt.scatter(coords[labels == i, 0], coords[labels == i, 1], label=str(i))
plt.legend()
plt.title("Clusters")
plt.show()
