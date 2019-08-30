import numpy as np
import joblib
from sklearn.model_selection import KFold

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

X = np.load('Banco de dados/Saidas/Num2250.npy')
y = np.load('Banco de dados/Saidas/Target2250.npy')

# Inversão para branco (255) no fundo preto (0) e normalização
X = X/255
X = -X + 1

# Necessário para treinar a rede neural convolucional
X = X.reshape(X.shape[0], 28, 28, 1)

# Análise com Cross Validation - CNN (Rede neural convolucional)
# 'analysing = True' para visualizar os scores, o que demora alguns segundos
# 'analysing = False' gera o classificador ótimo e salva

analysing = False

if analysing:
    kf = KFold(n_splits = 5)
    medias = []
    for i in range(1, 2):
        scores = []
        for train_index, test_index in kf.split(X):
            clf_cnn = Sequential()
            clf_cnn.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))
            clf_cnn.add(Conv2D(64, (3, 3), activation='relu'))
            clf_cnn.add(MaxPooling2D(pool_size=(2, 2)))
            clf_cnn.add(Dropout(0.25))
            clf_cnn.add(Flatten())
            clf_cnn.add(Dense(128, activation='relu'))
            clf_cnn.add(Dropout(0.5))
            clf_cnn.add(Dense(10, activation='softmax'))
            clf_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
            clf_cnn.fit(X_train, y_train, epochs=18, batch_size = 200)
            scores.append(clf_cnn.evaluate(X_test, y_test)[1])
        medias.append(np.mean(scores))
    print(medias)
else:
    # Gera o modelo com os parâmetros escolhidos com base na análise
    clf_cnn = Sequential()
    clf_cnn.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))
    clf_cnn.add(Conv2D(64, (3, 3), activation='relu'))
    clf_cnn.add(MaxPooling2D(pool_size=(2, 2)))
    clf_cnn.add(Dropout(0.25))
    clf_cnn.add(Flatten())
    clf_cnn.add(Dense(128, activation='relu'))
    clf_cnn.add(Dropout(0.5))
    clf_cnn.add(Dense(10, activation='softmax'))
    clf_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    clf_cnn.fit(X, y, epochs=18, batch_size = 200)

    # Salva o modelo
    joblib.dump(clf_cnn, 'Modelos_de_Classificacao/clf_cnn.pkl')
