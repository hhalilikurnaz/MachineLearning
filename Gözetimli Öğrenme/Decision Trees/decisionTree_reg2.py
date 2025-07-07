import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

#create dataset
X=np.sort(5* np.random.rand(80,1),axis=0)
y=np.sin(X).ravel()
y[::5] += 0.5*(0.5-np.random.rand(16))


plt.plot(X)
plt.show()

plt.scatter(X,y)
plt.show()


reg_1=DecisionTreeRegressor(max_depth=2)
reg_2=DecisionTreeRegressor(max_depth=5)
reg_1.fit(X,y)
reg_2.fit(X,y)

X_test=np.arange(0,5,0.5)[:,np.newaxis]
y_pred1=reg_1.predict(X_test)
y_pred2=reg_2.predict(X_test)


plt.figure()
plt.scatter(X,y,color='red',label='data')
plt.plot(X_test,y_pred1,color='blue',label='Max Depth: 2',linewidth=2)
plt.plot(X_test,y_pred1,color='green',label='Max Depth: 5',linewidth=2)
plt.xlabel('data')
plt.ylabel('target')
plt.legend()
plt.show()
'''
| Fonksiyon                 | Ne işe yarar                         |
| ------------------------- | ------------------------------------ |
| `plt.figure()`            | Yeni bir çizim başlatır (tuval açar) |
| `plt.scatter()`           | Veriyi nokta nokta çizer             |
| `plt.plot()`              | Tahminleri çizgiyle çizer            |
| `plt.xlabel() / ylabel()` | Eksenlere isim verir                 |
| `plt.legend()`            | Renkli açıklama kutuları gösterir    |
| `plt.show()`              | Tüm bu görseli ekrana bastırır       |
'''
'''
+----------------------------+
|         Target          |
|                            |
|   ● ● ● (scatter)          |
|    \       /               |
|     \_____/  (plot)        |
|                            |
|      Data (X axis)         |
+----------------------------+
'''