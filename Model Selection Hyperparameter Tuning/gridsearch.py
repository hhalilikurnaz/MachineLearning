from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np 

iris=load_iris()
X=iris.data
y=iris.target

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=42)

knn=KNeighborsClassifier()
knn_param_grid={"n_neighbors":np.arange(2,31)}

knn_grid_search=GridSearchCV(knn,knn_param_grid)
#knn algoritmamı grid searche göre fit ederek en iyi parametreleri bulmak 
knn_grid_search.fit(X_train,y_train)
print("KNN GRİD Seach Best Parameters",knn_grid_search.best_params_)
print("KNN GRİD Seach Best Accuracy",knn_grid_search.best_score_)


#Random SearchCrossValidation ile 

knn_random_search=RandomizedSearchCV(knn,knn_param_grid,n_iter=10)
#knn algoritmamı grid searche göre fit ederek en iyi parametreleri bulmak 
knn_random_search.fit(X_train,y_train)
print("KNN Random Seach Best Parameters",knn_random_search.best_params_)
print("KNN Random Seach Best Accuracy",knn_random_search.best_score_)

#grid seacrhte kaçtane hiperparametremiz varsa hepsini tek tek deniyor 
#parametre sayısı çoksa random searchle ile hemen buluyorsunuz .Hızlı çalışır ama en iyiyi bulamayabilir 

#Decision Tree # Grid Search
tree=DecisionTreeClassifier()
tree_param_grid={"max_depth":[3,5,7],
                 "max_leaf_nodes":[None,5,10,20,30]}

tree_grid_search=GridSearchCV(tree,tree_param_grid)
tree_grid_search.fit(X_train,y_train)
print("Tree  Grid Search Best Parameters :",tree_grid_search.best_params_)
print("Tree  Grid Search Best Accuracy",tree_grid_search.best_score_)

#Random Search
tree_random_search=RandomizedSearchCV(tree,tree_param_grid,n_iter=10)
tree_random_search.fit(X_train,y_train)
print("Tree RS Best Parameters :",tree_random_search.best_params_)
print("Tree RS Best Accuracy",tree_random_search.best_score_)
