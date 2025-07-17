from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
 
 # y=  a0 + a1x + -> linear regression
 #y= a0 + a1x1 + a2x2 + ---+anxn -> birden fazla bağımısz değişken bulunur 

X=np.random.rand(100,2)


coef=np.array([3,5]) #ger.ek doğrursal katsayılar bunlar 
y= 3+5*np.random.rand(100)+ np.dot(X,coef) # 0 yazdık bias yok 


fig=plt.figure()
ax=fig.add_subplot(111,projection="3d")
ax.scatter(X[:,0],X[:,1],y)
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("y")
plt.show()


lig_reg=LinearRegression()

lig_reg.fit(X,y)
fig=plt.figure()
ax=fig.add_subplot(111,projection="3d")
ax.scatter(X[:,0],X[:,1],y)
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("y")
plt.show()

x1,x2=np.meshgrid(np.linspace(0,1,10),np.linspace(0,1,10))
# tahmin düzlemini çizmek için yeni bir grid(ızgara) verisi oluşturuyor
#linspace (0-1 arasında 10 eşit sayı oluşturuyor )
y_pred=lig_reg.predict(np.array([x1.flatten(),x2.flatten()]).T)
#flatten 2d matristen 1d vektör yapar 
#T ile döndürüyoruz transpose 
ax.plot_surface(x1,x2,y_pred.reshape(x1.shape),alpha=0.3)
plt.title("Multi Variable linear regression")
print("Katsayılar",lig_reg.coef_)
print("Kesişim",lig_reg.intercept_)