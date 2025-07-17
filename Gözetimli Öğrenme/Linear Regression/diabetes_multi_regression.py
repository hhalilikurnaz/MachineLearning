from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np


diabetes=load_diabetes()

df=pd.DataFrame(data=diabetes.data,columns=diabetes.feature_names)
df['target']=diabetes.target
print(df.head(10))
X=df.drop(labels="bmi",axis=1)
y=df['bmi']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

lr=LinearRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)

mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
print("rmse:",rmse)