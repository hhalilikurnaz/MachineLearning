#Destek Vektör Makineleri
#Digit veri setini kullanıcaz (0-9) arası rakamlar
#support vector classification

from sklearn.datasets import load_digits
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report


data=load_digits()

df=pd.DataFrame(data.data)
df['target']=data.target
print(df.describe())
print(df.info())

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 5), subplot_kw={"xticks": [], "yticks": []})
 #subplots cünku tüm rakamları yazdırıcaz 10 tane rakamımız var ondan dolayı 2 satır 5 sutun yeter istesem 5 satır 2 sutunda yapabilirdim
for i,ax in enumerate(axes.flat):
    ax.imshow(data.images[i],cmap="binary",interpolation="nearest")
    ax.set_title(data.target[i])
plt.show()

X=df.drop(['target'],axis=1).values
y=df['target'].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

svm=SVC(kernel='linear',random_state=42)
svm.fit(X_train,y_train)
y_pred=svm.predict(X_test)

print(classification_report(y_test,y_pred))