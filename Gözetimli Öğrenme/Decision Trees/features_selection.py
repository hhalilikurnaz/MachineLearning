from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.inspection import DecisionBoundaryDisplay
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

data=load_iris()
df=pd.DataFrame(data=data.data,columns=data.feature_names)
print(df)
n_classes=len(data.target_names)
plot_colors="ryb"

for pairidx, pair in enumerate([(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]):


    X=data.data[:,pair]
    y=data.target

    clf=DecisionTreeClassifier().fit(X,y)


    ax=plt.subplot(2,3,pairidx+1)
    plt.tight_layout(h_pad=0.5,w_pad=0.5,pad=2.5) #=> Grafikleri sıkıştırır, aradaki boşlukları düzenler.
    DecisionBoundaryDisplay.from_estimator(clf,X,cmap=plt.cm.RdYlBu,response_method='predict',ax=ax,xlabel=data.feature_names[pair[0]],ylabel=data.feature_names[pair[1]])
    '''
    | Parametre                   | Ne işe yarıyor                            |
| --------------------------- | ----------------------------------------- |
| `clf`                       | Eğittiğimiz model                         |
| `X`                         | Girdi verisi                              |
| `cmap`                      | Renk haritası (Red-Yellow-Blue)           |
| `response_method='predict'` | Modelin tahmin sonucuna göre renk belirle |
| `ax=ax`                     | Hangi subplot üzerine çizilecek           |
| `xlabel`, `ylabel`          | X ve Y eksen isimleri (özellik isimleri)  |

    '''

for i,color in zip(range(n_classes),plot_colors):
    idx=np.where(y==i)
    plt.scatter(X[idx,0],X[idx,1],color=color,label=data.target_names[i],edgecolors='black')

plt.legend()
plt.show()
