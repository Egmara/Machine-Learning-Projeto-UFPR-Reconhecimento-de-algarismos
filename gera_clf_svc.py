import numpy as np
import joblib
from sklearn.model_selection import KFold
from sklearn.svm import SVC

X = np.load('Banco de dados/Saidas/Num2250.npy')
y = np.load('Banco de dados/Saidas/Target2250.npy')

# Inversão para branco (255) no fundo preto (0) e normalização
X = X/255
X = -X + 1

# Análise com Cross Validation - SVC
# 'analysing = True' para visualizar os scores, o que demora alguns segundos
# 'analysing = False' gera o classificador ótimo e salva

analysing = False

if analysing:
    kf = KFold(n_splits = 5)
    medias = []
    for C in [1, 2, 3, 4, 5]:
        scores = []
        clf_svc = SVC(kernel='rbf', gamma = 'auto', C = C)

        for train_index, test_index in kf.split(X):
            X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
            clf_svc.fit(X_train, y_train)
            scores.append(clf_svc.score(X_test, y_test))
        medias.append(np.mean(scores))
    print(medias)
else:
    # Gera o modelo com os parâmetros escolhidos com base na análise
    clf_svc = SVC(kernel='rbf', gamma = 'auto', C = 4)
    clf_svc.fit(X, y)

    # Salva o modelo
    joblib.dump(clf_svc, 'Modelos_de_Classificacao/clf_svc.pkl')
