import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


from ucimlrepo import fetch_ucirepo
#fetch dataset
heart_disease=fetch_ucirepo(id=45)
print(type(heart_disease.data))

df=pd.DataFrame(data=heart_disease.data.features)
df['target']=heart_disease.data.targets


print(df.isnull().sum()) # we have missing value
print(df.info())
print(df.describe())

df.dropna(inplace=True) #nan değerleri çıkarıyoruz 




X=df.drop(['target'],axis=1).values
y=df.target.values



#after droped N/A values
print(df.isnull().sum()) # we have missing value
print(df.info())
print(df.describe())

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=42)

lg=LogisticRegression(penalty='l2', C=1, solver='lbfgs', max_iter=100)

lg.fit(X_train,y_train)

accuracy=lg.score(X_test,y_test)
print("Logistic Regressin Accuracy:",accuracy)


#Lojistik regresyon normalde 2 li sınıflandırmalarda(binary classification) kullanılıyordu(sigmoid).5 tane sınıfı nasıl lojistik reg kullanarak yaptık.(softmax fonksiyonu ile  birlikte )