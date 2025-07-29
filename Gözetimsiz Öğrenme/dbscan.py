from sklearn.datasets import make_circles
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

#DBSCAN = Density-Based Spatial Clustering of Applications with Noise
#Yoğunluğa dayalı kümeleme algoritması

'''
Kalabalık grupları küme yapar.

Yalnızları gürültü/noise sayar
'''

'''
Bir pizzacıya gittin. Masalarda insanlar oturuyor.

Masada 4 kişi varsa → Bu masa bir kümeymiş, samimiler.

Masada tek kişi varsa, kimseyle konuşmuyorsa → Bu kişi noise, kendi halinde, dışlanmış olabilir.

Masada biri var ama etrafında başkaları da var, ama çok kalabalık değil → Sınır noktası olabilir.

DBSCAN diyor ki:

"Ben kalabalığı severim. Kim yalnız takılıyor, kim kalabalık, onu ayırırım!"
'''

X,_=make_circles(n_samples=1000,factor=0.5,noise=0.1,random_state=42)
plt.figure()
plt.scatter(X[:,0],X[:,1],c="blue")
plt.show()


dbscan=DBSCAN(eps=0.1,min_samples=5)
cluster_labels=dbscan.fit_predict(X)


plt.figure()
plt.scatter(X[:,0],X[:,1],c=cluster_labels,cmap="viridis")
plt.title("Result of DBSCAN")
plt.show()