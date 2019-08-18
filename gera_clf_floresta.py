import numpy as np
import joblib
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

X = np.load('Banco de dados/Saidas/Num2250.npy')
y = np.load('Banco de dados/Saidas/Target2250.npy')

# Inversão para branco (255) no fundo preto (0) e normalização
X = X/255
X = -X + 1

# Análise com Cross Validation - Floresta
# 'analysing = True' para visualizar os scores, o que demora alguns segundos
# 'analysing = False' gera o classificador ótimo e salva

analysing = True

if analysing:
    kf = KFold(n_splits = 5)
    medias = []
    for max_depth in range(8,14):
        scores = []
        clf_floresta = RandomForestClassifier(max_depth = max_depth, n_estimators = 75)

        for train_index, test_index in kf.split(X):
            X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
            clf_floresta.fit(X_train, y_train)
            scores.append(clf_floresta.score(X_test, y_test))
        medias.append(np.mean(scores))
    print(medias)
else:
    # Gera o modelo com os parâmetros escolhidos com base na análise
    clf_floresta = RandomForestClassifier(max_depth = 12, n_estimators = 75)
    clf_floresta.fit(X, y)

    # Salva o modelo
    joblib.dump(clf_floresta, 'Modelos_de_Classificacao/clf_floresta.pkl')
