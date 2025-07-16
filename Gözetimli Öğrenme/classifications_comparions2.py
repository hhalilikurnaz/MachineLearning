import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import DecisionBoundaryDisplay

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Veri setlerini oluştur
X1, y1 = make_classification(n_features=2, n_redundant=0, n_informative=2,
                             n_clusters_per_class=1, random_state=42)
X1 += np.random.uniform(size=X1.shape)  # noise ekle

X2, y2 = make_moons(noise=0.3, random_state=42)
X3, y3 = make_circles(noise=0.2, factor=0.5, random_state=42)

datasets = [(X1, y1), (X2, y2), (X3, y3)]

# Model isimleri ve classifier'lar
names = ["Nearest Neighbors", "Linear SVM", "Decision Tree", "Random Forest", "Naive Bayes"]
classifiers = [
    KNeighborsClassifier(),
    SVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GaussianNB()
]

figure = plt.figure(figsize=(15, 9))

# Renk paleti
cm = plt.cm.RdBu
cm_bright = ListedColormap(["#FF0000", "#0000FF"])

# Her veri seti için döngü
for ds_index, (X, y) in enumerate(datasets):
    # Eğitim ve test verisini ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # İlk sütuna (input data) çizim
    ax = plt.subplot(len(datasets), len(classifiers)+1, ds_index*(len(classifiers)+1) + 1)
    if ds_index == 0:
        ax.set_title("Input data")
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors='k')
    ax.set_xticks(())
    ax.set_yticks(())

    # Her classifier için döngü
    for clf_index, (name, clf) in enumerate(zip(names, classifiers)):
        ax = plt.subplot(len(datasets), len(classifiers)+1, ds_index*(len(classifiers)+1) + clf_index + 2)
        model = make_pipeline(StandardScaler(), clf)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)

        # Karar sınırını çiz
        DecisionBoundaryDisplay.from_estimator(model, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5)

        # Eğitim ve test noktalarını tekrar çiz
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors='k')

        ax.set_xticks(())
        ax.set_yticks(())
        if ds_index == 0:
            ax.set_title(name)
        ax.text(X[:, 0].max() - 0.3, X[:, 1].min() + 0.3, f"{score:.2f}", size=9, horizontalalignment='right')

plt.tight_layout()
plt.show()
