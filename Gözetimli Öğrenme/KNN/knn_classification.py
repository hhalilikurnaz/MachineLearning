#(1) Veri Setleri incelemesi
#sklearn:ML Library


from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
cancer=load_breast_cancer()
df=pd.DataFrame(data=cancer.data,columns=cancer.feature_names)
df['target']=cancer.target

#EDA Görselleştirmeli farklı analizli veri analizi demek



#(2) Makine ogrenemsi modelinin seçilmesi-KNN Sınfılandırıcı
X=cancer.data #features
y=cancer.target #target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

#olceklendirme preprocessing
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)



#knn modeli oluştur ve train et
knn=KNeighborsClassifier(n_neighbors=3) #Model Olusturuyoruz. komsu parametresini unutma ****
knn.fit(X_train,y_train) #features ve target lazım bize #fit fonkisyonu verimizi(samples+taget) kullanarak knn algoritmasını eğitir



#(3) modelin train elde edilmesi



#(4) sonucların değerlendirlmesi
y_prediction=knn.predict(X_test) # bu mantıklı dğeiş su an
print(y_prediction)

accuracy=accuracy_score(y_test,y_prediction)
print("Doğruluk:",accuracy)

confusion_matrix=confusion_matrix(y_test,y_prediction)
print(confusion_matrix)
#(5) hiperparametre ayarlanması
#temel amaçlarımdan ilki accuracy arttırmak
'''
KNN:Hyperparametr = K
K:1,2,3,....N
Accuracy:%a,%b,%c,%d,...

'''
accuracy_values=[]
k_values=[]
for k in range(1,21):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_prediction=knn.predict(X_test)
    accuracy=accuracy_score(y_test,y_prediction)
    accuracy_values.append(accuracy)
    k_values.append(k)
    print(accuracy_values)
    print(k_values)

plt.figure()
plt.plot(k_values,accuracy_values,marker="o",linestyle="-")
plt.title("K Değerine Göre accuracy")
plt.xlabel("K Değeri"),
plt.ylabel("Accuracy")
plt.xticks(k_values)
plt.grid(True)
plt.show()




