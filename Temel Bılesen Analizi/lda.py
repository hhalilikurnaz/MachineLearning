from sklearn.datasets import fetch_openml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt


mnist=fetch_openml("mnist_784",version=1)
#70.k veri var 784 feature var 28*28 lik 
#etiketler rakam(0-9 ) arası

X=mnist.data
y=mnist.target.astype(int)

lda=LinearDiscriminantAnalysis(n_components=2) #784 featuresi 2 feature düşürdüm 
X_lda=lda.fit_transform(X,y)


plt.figure()
plt.scatter(X_lda[:,0],X_lda[:,1],c=y,cmap="tab10",alpha=0.6)
plt.title("LDA of MNIST Dataset")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.colorbar(label="Digits")
plt.show()


