import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
X=np.sort(5*np.random.rand(40,1),axis=0) #uniform dağılım ile oluşturuldu.featurlerim
y=np.sin(X).ravel() #target #=>(2D’yi 1D yapar, özellikle y değişkeni için şart)

#add noise 
y[::5] += 1*(0.5-np.random.rand(8))
plt.scatter(X,y)
#kendimize x değişkeni yaptık numpy ile random sayılarla 
#y değişkeni içinde x in sinüsünü aldık sonra 5 sıraya bir ekleme yaptık(biraz zor olsun diye)
#plt.show()

T=np.linspace(0, 5, 500)[:,np.newaxis] #=>0 ile 5 arasında 500 tane sayı üret (eşit aralıklı).
#=>Bu listeyi 2D şekle sok. Çünkü KNNRegressor 2D array ister.
#Yani: (500,) → (500,1) haline getiriliyor.

for i,weight in enumerate (['uniform','distance']):
     #=>weights="uniform": Her komşu eşit ağırlıkta (dilersen "distance" de diyebilirsin → uzak olan daha az etki eder)
    knn=KNeighborsRegressor(n_neighbors=5,weights=weight)
    y_pred=knn.fit(X,y).predict(T)

    plt.subplot(2,1,i+1)
    plt.scatter(X,y,color="green",label='data')
    plt.plot(T,y_pred,color="blue",label="prediction")
    plt.axis=("tight") #=> # Grafik kenar boşluklarını sıkar
    plt.legend() #=># Etiketleri göster ("data", "prediction")
    plt.title("KNN Regressor weights = {}".format(weight))
plt.tight_layout() #=>: Alt alta iki subplot düzgün otursun diye boşluk ayarlaması yapar.
plt.show()
