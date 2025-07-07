from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt

import pandas as pd

#Veri seti inceleme
iris=load_iris()

df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['target']=iris.target
print(df)

X=iris.data
y=iris.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#DT modeli oluştur ve train et
tree=DecisionTreeClassifier(criterion="gini",max_depth=5,random_state=42) #criterion="entropy" de olabilir 
#gini=> temiz bölünme istiyorum.->Hız istiyorsan
#entropy=>bilgi kazancına bakacağım.->bilgi teorisi temelli bir analiz istiyorsan
tree.fit(X_train,y_train)

#DT Evaluation
y_pred=tree.predict(X_test)
accuracy_score=accuracy_score(y_test,y_pred)

print("Doğruluk",accuracy_score)

confusion_matrix=confusion_matrix(y_test,y_pred)
print(confusion_matrix)

#tree'yi görselleştireceğiz
plt.figure(figsize=(16,10))
plot_tree(tree,filled=True,feature_names=iris.feature_names,class_names=iris.target_names)
plt.show()

#En yukardaki feature en önemli featuredır bizim için
feature_importances=tree.feature_importances_
feature_names=iris.feature_names

feature_importances_sorted=sorted(zip(feature_importances,feature_names))
#zip fonksiyonu =>→ Her feature importance değerini, ilgili feature adıyla eşleştiriyor.
#sorted fonksiyonu =>Bu da bu eşleşmeleri önem değerine göre küçükten büyüğe sıralıyor.

for importance,feature_name in feature_importances_sorted:
    print(f"{feature_name}:{importance}")

