# gera resultados para algumas imagens de teste

# carrega os classificadores salvos
import joblib
clf_redeneural = joblib.load('Modelos_de_Classificacao/clf_redeneural.pkl')
clf_floresta = joblib.load('Modelos_de_Classificacao/clf_floresta.pkl')

modelos = {'rede_neural': clf_redeneural, 'floresta': clf_floresta}

from classifica_teste import classifica_teste

# Teste 1
resultados = classifica_teste('testes/teste1.jpeg', modelos)
print(f'Rede Neural: {resultados[0]}, \nFloresta: {resultados[1]}')
print('Gabarito do teste 1: 2, 5, 5\n')

# Teste 2
resultados = classifica_teste('testes/teste2.jpeg', modelos)
print(f'Rede Neural: {resultados[0]}, \nFloresta: {resultados[1]}')
print('Gabarito do teste 2: 0, 2, 3, 7, 4, 1\n')

# Teste 3
resultados = classifica_teste('testes/teste3.jpeg', modelos)
print(f'Rede Neural: {resultados[0]}, \nFloresta: {resultados[1]}')
print('Gabarito do teste 3: 5, 0, 2, 1, 3, 8, 7, 6, 9, 4\n')

# Teste 4
resultados = classifica_teste('testes/teste4.jpeg', modelos)
print(f'Rede Neural: {resultados[0]}, \nFloresta: {resultados[1]}')
print('Gabarito do teste 4: 0, 1, 9, 2, 3, 5, 6, 8, 4\n')

#Teste 5
resultados = classifica_teste('testes/teste5.jpeg', modelos)
print(f'Rede Neural: {resultados[0]}, \nFloresta: {resultados[1]}')
print('Gabarito do teste 5: 4, 2, 5, 1, 7, 0, 8, 9\n')

# Teste 6
resultados = classifica_teste('testes/teste6.jpeg', modelos)
print(f'Rede Neural: {resultados[0]}, \nFloresta: {resultados[1]}')
print('Gabarito do teste 6: 2, 9, 5, 4, 7\n')
