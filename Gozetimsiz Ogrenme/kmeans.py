
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs #veri seti oluşturma kütüphanesi
import matplotlib.pyplot as plt


X,_=make_blobs(n_samples=300,centers=4,cluster_std=0.6,random_state=42) # _ demek ne oldugu önemli değil kullanmıcaz demek
plt.scatter(X[:,0],X[:,1])
#cluster_std=0.6 → Kümelerin yayılımı (ne kadar dağınık?)

plt.title("Örnek Veri")
plt.show()

kmeans=KMeans(n_clusters=4)
kmeans.fit(X)
labels=kmeans.labels_
#kümelerin merkezini çizdirecez

plt.figure()
plt.scatter(X[:,0],X[:,1],c=labels,cmap="viridis")
plt.show()

centers=kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X')
plt.show()

