from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold

#X = np.load('Banco de dados/Saidas/Num1650.npy')
#y = np.load('Banco de dados/Saidas/Target1650.npy')
X = np.load('Banco de dados/Saidas/Num2250.npy')
y = np.load('Banco de dados/Saidas/Target2250.npy')

# inversao para branco (255) no fundo preto (0) e normalização
X = X/255
X = -X + 1

kf = KFold(n_splits = 5)
kf.get_n_splits(X)

# analise - Floresta
medias = []
for i in range(3,7,1):

    scores = []
    clf_floresta = RandomForestClassifier(max_depth = i, n_estimators = 10)

    for train_index, test_index in kf.split(X):

        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        clf_floresta.fit(X_train, y_train)
        scores.append(clf_floresta.score(X_test, y_test))

    medias.append(np.mean(scores))

#print(medias)

# modelo otimo
clf_floresta = RandomForestClassifier(max_depth = 7, n_estimators = 10)
clf_floresta.fit(X, y)

# salva o modelo
joblib.dump(clf_floresta, 'Modelos_de_Classificacao/clf_floresta.pkl')
