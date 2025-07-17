from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt

diabetes_X, diabetes_y = load_diabetes(return_X_y=True)

df = pd.DataFrame(data=diabetes_X, columns=load_diabetes().feature_names)
df['target'] = diabetes_y

print(df.head())

X = df[["bmi"]]
y = df["target"]

X_train = X[:-20]
X_test = X[-20:]

y_train = y[:-20]
y_test = y[-20:]

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred = lin_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R²:", r2)
print("r2",r2)

plt.scatter(X_test,y_test,color="black",label="Gerçek") #noktacıklar koyar 
plt.plot(X_test,y_pred,color="blue",label="Tahmin") #doğru şeklinde çizer
plt.xlabel("BMI") 
plt.ylabel("Target")
plt.title("Linear Regression - Gerçek vs Tahmin")
plt.legend()
plt.show()

"""
| Komut      | Ne işe yarar?                                              |
| ---------- | ---------------------------------------------------------- |
| `xlabel()` | Eksenin altına yazı koyar (açıklama)                       |
| `xticks()` | X ekseninde hangi değerlerin görüneceğini ayarlar (etiket) |

"""