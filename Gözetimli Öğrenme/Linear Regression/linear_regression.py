#bağımsız değişkenlerle (X) bağımlı değişken (Y) arasındaki ilişkiyi bir doğru denklemine oturtarak açıklayan bir regresyon modelidir.
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

#veri oluştur 
X=np.random.randn(100,1)
y= 3 + (4* X)+ np.random.randn(100,1)

plt.scatter(X,y)
plt.show()


lin_reg=LinearRegression()
lin_reg.fit(X,y)

plt.figure()
plt.scatter(X,y)
plt.plot(X,lin_reg.predict(X),color="red",alpha=0.7)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Lineer Regresyon")
plt.show()

a0=lin_reg.coef_[0] #Bu, modelin öğrendiği katsayıları (coefficient) verir.
print(a0)
a1=lin_reg.coef_[0][0]

for i in range(100):
    y_=a0 +a1 *X
    plt.plot(X,y_,color="green",alpha=0.7)

