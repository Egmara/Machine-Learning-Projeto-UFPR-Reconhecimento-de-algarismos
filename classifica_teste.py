# Esta função classifica os algarismos de uma imagem de teste

def classifica_teste(filename, modelos):

    import numpy as np

    # carrega funcao que separa algarismos da imagem
    from separa_algarismos import separa_algarismos

    # chama a funcao separa_algarismos
    Vetores = separa_algarismos(filename)

    # forma padrão
    for i in np.arange(len(Vetores)):
        Vetores[i] = (-Vetores[i] + 255)/255

    # classifica
    resultados = []
    resultados.append(modelos['rede_neural'].predict(Vetores[0:]))
    resultados.append(modelos['floresta'].predict(Vetores[0:]))
    resultados.append(modelos['svc'].predict(Vetores[0:]))
    # cnn outputs array with probabilities for each class
    # this needs to be converted to the actual prediction
    cnn_results = resultados[0]
    X = np.asarray(Vetores[0:])
    X = np.reshape(X, (X.shape[0], 28, 28, 1))
    for i in range(0, X.shape[0]):
        cnn_results[i] = np.argmax(modelos['cnn'].predict(X[i:i+1]))
    resultados.append(cnn_results)

    return resultados
