from sklearn.datasets import load_iris
from sklearn.decomposition import PCA 

#PCA Temel Bileşen Analizi bir boyut indirgeme tekniklerinden birisidir
#Parametresi n_components elimizdeki veri setini kaç boyuta indireyim
import matplotlib.pyplot as plt

iris=load_iris()
X=iris.data
y=iris.target

pca=PCA(n_components=2) # 2 adet princible components  (temel bileşen)

X_pca=pca.fit_transform(X)

print(X_pca) #iris verisetimin 2 temel bileşeni

plt.figure()
for i in range(len(iris.target_names)):
    plt.scatter(X_pca[y==i,0],X_pca[y==i,1],label=iris.target_names[i])

plt.xlabel("PC1")
plt.ylabel("Pc2")
plt.title("PCA of iris Dataset")
plt.legend()
plt.show()
#%%
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt

iris=load_iris()
X=iris.data
y=iris.target

pca=PCA(n_components=3) # 2 adet princible components  (temel bileşen)

X_pca=pca.fit_transform(X)

fig=plt.figure(1,figsize=(8,6))
ax=fig.add_subplot(111,projection="3d",elev=-150,azim=110)
ax.scatter(X_pca[:,0],X_pca[:,1],X_pca[:,2],c=y,s=40)
ax.set_title("First Three PCA Dimensions of iris Dataset")
ax.set_xlabel("1st Eigenvector")
ax.set_ylabel("2st Eigenvector")
ax.set_zlabel("3st Eigenvector")
plt.legend()
plt.show()


