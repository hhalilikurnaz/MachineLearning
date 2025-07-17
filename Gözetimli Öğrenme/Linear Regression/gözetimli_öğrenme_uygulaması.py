from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import pandas as pd 
import numpy as np 


data=fetch_california_housing()
print(type(data))

df=pd.DataFrame(data=data.data,columns=data.feature_names)
df['Target']=data.target
print(df.head(10))
print(df['Target'])


X=df.drop(labels="Target",axis=1)
y=df['Target']


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

poly_feat=PolynomialFeatures(degree=2)
X_train_poly=poly_feat.fit_transform(X_train)
X_test_poly=poly_feat.transform(X_test)


pr = LinearRegression()
pr.fit(X_train_poly, y_train)

y_pred = pr.predict(X_test_poly)  

mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)

print("Polynomial Regression rmse",rmse)

lr=LinearRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
print("Linear Regression rmse",rmse)
