import numpy as np
import joblib
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier

X = np.load('Banco de dados/Saidas/Num2250.npy')
y = np.load('Banco de dados/Saidas/Target2250.npy')

# Inversão para branco (255) no fundo preto (0) e normalização
X = X/255
X = -X + 1

# Análise com Cross Validation - Rede Neural
# 'analysing = True' para visualizar os scores, o que demora alguns segundos
# 'analysing = False' gera o classificador ótimo e salva

analysing = True

if analysing:
    kf = KFold(n_splits = 5)
    medias = []
    for layer1_size in range(50, 126, 25):
        scores = []
        clf_redeneural = MLPClassifier(hidden_layer_sizes=(layer1_size), alpha=1e-4, solver='lbfgs', verbose=False)

        for train_index, test_index in kf.split(X):
            X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
            clf_redeneural.fit(X_train, y_train)
            scores.append(clf_redeneural.score(X_test, y_test))
        medias.append(np.mean(scores))
    print(medias)
else:
    # Gera o modelo com os parâmetros escolhidos com base na análise
    clf_redeneural = MLPClassifier(hidden_layer_sizes=(75), alpha=1e-4, solver='lbfgs', verbose=False)
    clf_redeneual.fit(X, y)

    # Salva o modelo
    joblib.dump(clf_redeneural, 'Modelos_de_Classificacao/clf_redeneural.pkl')
