from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score


data=fetch_california_housing()

df=pd.DataFrame(data=data.data,columns=data.feature_names)
print(df.head(4))
print(df.describe())

print(df.info())

X=data.data
y=data.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


mse_list=[]


for n in range(1, 100, 20):
    rf = RandomForestRegressor(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    mse =mean_squared_error(y_test, y_pred)
    mse_list.append((n, mse))

for i, (n, mse) in enumerate(mse_list, start=1):
    print(f"{i:02}. n_estimators = {n:<3} → MSE = {mse:.3f}")


