from sklearn.datasets import make_classification,make_moons,make_circles
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.inspection import DecisionBoundaryDisplay

from sklearn.svm import SVC

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


X,y=make_classification(n_features=2,n_redundant=0,n_informative=2,n_clusters_per_class=1,random_state=42) #redundant bilgi içermeyen featureslar
# n_features=2: 2 özellikli veri kümesi oluşturur (genellikle x ve y koordinatları gibi).
# n_redundant=0: Hiçbir gereksiz özellik olmaz (diğer özelliklerden türetilebilen özellikler).
# n_informative=2: Tüm özellikler sınıflandırma için bilgi içerir.
# n_clusters_per_class=1: Her sınıf için tek bir küme oluşturulur.
# random_state=42: Tekrarlanabilir sonuçlar için rastgelelik tohumu

X += np.random.uniform(size=X.shape) #adding noise
Xy=(X,y)#tupple


'''


plt.scatter(X[:,0],#tüm satırların ilk sutunundan veri alır demek
            X[:,1],#tüm satırların 2.sutunundan veri alır
            c=y)#noktaları y değerlerie göre renklendir demek





X,y=make_moons(noise=2.2,random_state=42)
plt.scatter(X[:,0],#tüm satırların ilk sutunundan veri alır demek
            X[:,1],#tüm satırların 2.sutunundan veri alır
            c=y)#noktaları y değerlerie göre renklendir demek


plt.show()

X,y=make_circles(noise=0.1,factor=0.3,random_state=42)
plt.scatter(X[:,0],
            X[:,1],
            c=y)
plt.show()
'''


datasets=[Xy,
          make_moons(noise=2.2,random_state=42),# Ay şeklindeki veri kümesi, biraz gürültülü
          make_circles(noise=0.3,factor=0.3,random_state=42)# İç içe daireler şeklindeki veri kümesi, gürültülü
          ]

fig=plt.figure(figsize=(6,9))
i=1

for ds_cnt,ds in enumerate(datasets):
    X,y=ds  # Veri kümesi (X: özellikler, y: etiketler) ayrıştırılır.
    if ds_cnt==0:
        colors='darkred' # İlk veri kümesi için renk
    elif ds_cnt==1:
        colors='darkblue'  # İkinci veri kümesi için renk
    else:
        colors='darkgreen'# Diğerleri için renk (burada sadece üçüncü veri kümesi)

    ax=plt.subplot(len(datasets),1,ds_cnt+1)
    ax.scatter(X[:,0],
               X[:,1],
               c=y, # Noktaların rengi y değerlerine göre belirlenir (sınıflandırma etiketleri).
               cmap=plt.cm.coolwarm,edgecolors='black')
plt.show()

names=["Nearest Neighors","Linear SVM","Decision Tree","Random Forest","Naive Bayes"]

classifiers=[KNeighborsClassifier(),
             SVC(),
             DecisionTreeClassifier(),
             RandomForestClassifier(),
             GaussianNB()]

fig=plt.figure(figsize=(6,9))
i=1  #subplot sayacı

for ds_count,ds in enumerate(datasets):
    X,y=ds # Veri kümesi (X, y) ayrıştırılır.
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

    cm_bright=ListedColormap(["darkred","darkblue"])


    ##----Input Data Sutunu
    ax=plt.subplot(len(datasets),len(classifiers)+1 ,i)
    
    if ds_cnt == 0: # Sadece ilk veri kümesi için "INPUT DATA" başlığı ayarlanır.
        ax.set_title("INPUT DATA")


    #plot training data 
    ax.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap =cm_bright ,edgecolors="black")      
      #plot testing data 
    ax.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap =cm_bright ,edgecolors="black",alpha=0.6)     

    i+=1

    #Sınıflandırıcıları çalıştırıyoruz 
    #----Modeller
    for names,clf in zip(names,classifiers):

        ax=plt.subplot(len(datasets),len(classifiers)+1 ,i)


        #farklı veri setleri ve sınıflandırıcılar arasında dengeyi sağlıyabilmek için standarizyon yapmamız lazım
        #sklearnden piple  oluşturup bunu bizim için yaptırıcaz


        # Pipeline: Scale → Model
        
        
        clf=make_pipeline(StandardScaler(),clf)
        clf.fit(X_train,y_train)
        #x ve y train ölce standart hala geliyor sonra fitting işlemi yapılıyor 

        score=clf.score(X_test,y_test)  #accuracy

        # Karar sınırı
        DecisionBoundaryDisplay.from_estimator(clf,X,cmap=plt.cm.RdBu,alpha=0.7,ax=ax,eps=0.5)
        
        #-----Veri Noktalarını Tekrar Çiz----
        #plot training data 
        ax.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap =cm_bright ,edgecolors="black")      
        #plot testing data 
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors='k')

        if ds_cnt==0:

            ax.set_title(names)

        ax.text(
            X[:,0].max() -0.15,
            X[:,1].min() +0.20,
            str(score))

        i +=1 

plt.subplots_adjust()
plt.show()






