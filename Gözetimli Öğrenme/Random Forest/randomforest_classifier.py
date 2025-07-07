#insan verilerini kullanıcaz

import pandas as pd
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

oli=fetch_olivetti_faces()


df=pd.DataFrame(data=oli.data)
df.info()
df.describe()
df.isnull().sum()

plt.figure()
for i in range(2):
    plt.subplot(1,2,i+1)
    plt.imshow(oli.images[i],cmap='gray')
    plt.axis('off')

plt.show()

X=oli.data
y=oli.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

rf=RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(X_train,y_train)

y_pred=rf.predict(X_test)

accuracy_score2=accuracy_score(y_test,y_pred)
confusion_matrix=confusion_matrix(y_test,y_pred)

print("Doğruluk:",accuracy_score2)

accuracy_value = []

for n in range(1, 201, 10):
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)         
    accuracy_value.append((n, acc))             
    

# enumerate ile yazdır
for i, (n, acc) in enumerate(accuracy_value, start=1):
    print(f"{i:02}. n_estimators = {n:<3} → Accuracy = {acc:.3f}")

# En iyi sonucu bul
best_index, (best_n, best_acc) = max(enumerate(accuracy_value), key=lambda x: x[1][1])
print(f"\n En yüksek doğruluk: #{best_index+1} → {best_n} ağaç ile {best_acc:.3f}")


    
