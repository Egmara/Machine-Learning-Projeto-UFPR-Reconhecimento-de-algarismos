from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold

X = np.load('Banco de dados/Saidas/Num2250.npy')
y = np.load('Banco de dados/Saidas/Target2250.npy')

# inversao para branco (255) no fundo preto (0) e normalização
X = X/255
X = -X + 1

kf = KFold(n_splits = 5)
kf.get_n_splits(X)

# analise - Rede Neural
medias = []
for i in range(50,201,25):

    scores = []
    clf_redeneural = MLPClassifier(hidden_layer_sizes=(i),activation='relu', max_iter=55, alpha=1e-4,
                    solver='lbfgs', verbose=False, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

    for train_index, test_index in kf.split(X):

        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        clf_redeneural.fit(X_train, y_train)
        scores.append(clf_redeneural.score(X_test, y_test))

    medias.append(np.mean(scores))

    print(medias)

# modelo otimo
#clf_redeneural = MLPClassifier(hidden_layer_sizes=(100),activation='relu', max_iter=55, alpha=1e-4,
#                solver='lbfgs', verbose=False, tol=1e-4, random_state=1,
#                learning_rate_init=.1)
#clf_redeneual.fit(X, y)

# salva o modelo
#joblib.dump(clf_redeneural, 'Modelos_de_Classificacao/clf_redeneural.pkl')
