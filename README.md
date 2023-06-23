# World-Population
 Analise da taxa de fertilidade historica de todos os paises 

Projeto World Population - Forecast Dataset (1955-2050)

Contexto:
Foi obtido uma base dados contém informações sobre os dados históricos e atuais (1955-2020) das populações de todos os países.

Objetivo:
Construir um modelo de previsao da taxa de fertilidade e sua relacao com variaveis, visto que a mesma influencia diretamente no total da populacao mundial e nas politicas do paises.

As bases de dados foram retiradas do site kaggle: https://www.kaggle.com/datasets/anxods/world-population-and-forecast-dataset

Extracao de dados:
No nosso projeto foi obtido os dados descritos no link acima, onde os mesmo foram ajustados (limpeza dos dados), em seguida foi realizado uma analise exploratoria de forma dinamica e visual atraves de graficos e valores.
Apos a analise exploratoria foi feito a modelagem para metrica no calculo do modelo de previsao. Assim foi escolhido tres modelos a serem testados (RandomForest, LinearRegression e Extra Trees). Em seguida foi verificado atraves dos resultados que o modelo Extra Trees apresetaram o melhor resultado de R2 (proporcao da variancia da variavel independente pelo modelo) e RSME (mensuracao do desvio padrao residual), ao final foi apresentado a relacao de cada variavel com o da taxa de fertilidade.