from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

iris=load_iris()
X=iris.data
y=iris.target

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=42)

#Decision Tree # Grid Search
tree=DecisionTreeClassifier()
tree_param_grid={"max_depth":[3,5,7],
                 "max_leaf_nodes":[None,5,10,20,30]}
nb_cv=3
tree_grid_search=GridSearchCV(tree,tree_param_grid)
tree_grid_search.fit(X_train,y_train)
print("Tree  Grid Search Best Parameters :",tree_grid_search.best_params_)
print("Tree  Grid Search Best Accuracy",tree_grid_search.best_score_)


for mean_score, params in zip(tree_grid_search.cv_results_["mean_test_score"],tree_grid_search.cv_results_["params"]):
    print(f"Ortalama Test skoru : {mean_score},Parametreler {params}")

cv_result=tree_grid_search.cv_results_
for i ,params in enumerate((cv_result["params"])):
    print(f"Parametreler : {params}")
    for j in range(nb_cv):
        accuracy=cv_result[f"split{j}_test_score"][i]
        print(f"\t fold {j+1} - Accuracy : {accuracy}")