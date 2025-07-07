from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,root_mean_squared_error
import numpy as np
import pandas as pd


data=load_diabetes()

df=pd.DataFrame(data=data.data,columns=data.feature_names)
print(df)

X=data.data
y=data.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Model

tree_reg=DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train,y_train)

y_prediction=tree_reg.predict(X_test)

mse=mean_squared_error(y_test,y_prediction)
print(mse)

rmse=np.sqrt(mse)
print(rmse)
#Olceklendirme yapıldıgı için veriler - şekilde yazılmıs
