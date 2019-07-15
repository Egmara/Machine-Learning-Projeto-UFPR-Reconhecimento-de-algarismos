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

    return resultados
