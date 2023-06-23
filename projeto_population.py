#!/usr/bin/env python
# coding: utf-8

# ### World Population & Forecast Dataset (1955-2050)
# 
# - Projeto iniciado com o objetivo de estimar a densidade urbana prevista baseado na variavel de taxa de fertilidade.
# 

# ### Importacao das bibliotecas e Bases de dados

# In[2]:


import pandas as pd 
import pathlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn.objects as so
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split


# - Verificado que a base de dados estava separada por pais, entao foi feito o procedimento de condesacao das bases de dados em um data frame

# In[3]:


base_population = pd.DataFrame()
caminho_bases = pathlib.Path('population\countries')

for arquivo in caminho_bases.iterdir():
    
    df = pd.read_csv(caminho_bases / arquivo.name)
    base_population = base_population.append(df)

display (base_population)


# - Exclusao das linhas que contenham informacoes em branco

# In[4]:


base_population.dropna(inplace=True)


# In[5]:


base_population.set_index('Urban  Pop %', inplace=True)
base_population = base_population.drop(axis=0, index='N.A.')
base_population.reset_index(inplace=True)

base_population


# - Verificacao dos tipos dos dados em cada coluna e em seguida foi transformacao dos mesmos em numerico

# In[6]:


print(base_population.dtypes)


# In[7]:


base_population['Urban  Pop %'] = base_population['Urban  Pop %'].str.replace('%','')
base_population['Yearly %   Change'] = base_population['Yearly %   Change'].str.replace('%','')
base_population["Country's Share of  World Pop"] = base_population["Country's Share of  World Pop"].str.replace('%','')

base_population['Urban  Pop %'] = base_population['Urban  Pop %'].astype(np.float32) /100
base_population['Yearly %   Change'] = base_population['Yearly %   Change'].astype(np.float32) /100
base_population["Country's Share of  World Pop"] = base_population["Country's Share of  World Pop"].astype(np.float32) /100
base_population["Urban Population"] = base_population["Urban Population"].astype(np.int64) /100

display(base_population)    


# In[8]:


df_agrupado = 0
df_agrupado2 = 0 
lista = []

df_agrupado = base_population.loc[base_population['Year']==2020,:]
df_agrupado = df_agrupado.sort_values(by=['Fertility Rate'],ascending=True)
lista = df_agrupado.iloc[0:6,1].to_list()
df_agrupado2 = base_population.loc[base_population['country'].isin(lista),:]
df_agrupado2


# - Novamente apesar de nao utilizarmos a coluna de descricao dos paises na nossa analise estatistica, seria relevando analisar os seis princiais paises que apresentaram queda na taxa de fertilidade.

# In[9]:


so.Plot(df_agrupado2, x="Year", y="Fertility Rate").facet(col="country", wrap=3).add(so.Line())


# In[10]:


so.Plot(df_agrupado2, x="Year", y='Median Age',group='country').add(so.Line())


# - A coluna "Rank" sera excluida do modelo pois e uma classificacao arbitraria da base de dados, onde nao ira agregar informacoes relevantes ao modelo
# - A coluna de paises sera excluida pois a analise nao se propoe a realizar uma analise plitica/ geografica, visto que sera realizado testes estatisticos levando em conta dados numericos objetivos, apesar de gerarmos mais a frente a titulo de visualicao geral, graficos com a taxa de fetilidade de alguns paises.

# In[11]:


base_population.drop('Rank',axis=1,inplace=True)
base_population.drop('country',axis=1,inplace=True)


# - A base de dados sera tratada sera salva em csv para novas analises futuras

# In[12]:


base_population.to_csv('base_consolidada2.csv', sep=',')


# In[13]:


print(base_population.dtypes)


# ### Análise Exploratória e Tratar Outliers

# - Nessa etapa sera gerado primeiro um grafico de calor, para mapear de forma visual a relacao das variaveis entre si

# In[14]:


def mapa_calor(x,y,escala):
    sns.set_theme()
    b = pd.DataFrame(x,y)
    ax = plt.subplots(figsize=(15,10))
    sns.heatmap(base_population.corr(), annot=True)
   


# In[15]:


mapa_calor(base_population['Year'],base_population['Population'],base_population['Fertility Rate'])


# ### Definicao de funcoes para a analise de outliers
# 

# - Definicao de algumas funcoes para ajudar na analise de outliers

# In[16]:


def limites(coluna):
    q1 = coluna.quantile(0.25)
    q3 = coluna.quantile(0.75)
    amplitude = q3 - q1
    return q1 - 1.5 * amplitude, q3 + 1.5 * amplitude

def excluir_outliers(df, nome_coluna):
    qtdelinhas = df.shape[0]
    lim_inf, lim_sup = limites(df[nome_coluna])
    df = base_population.loc[(base_population[nome_coluna] >= lim_inf) & (base_population[nome_coluna] <= lim_sup), :]
    linhas_removidas = qtde_linhas - df.shape[0]
    return df, linhas_removidas


# In[17]:


print(limites(base_population['Fertility Rate']))
base_population['Fertility Rate'].describe()


# In[18]:


def diagrama_caixa(coluna):
    fig, (ax1,ax2) = plt.subplots(1,2)
    fig.set_size_inches(15,5)
    sns.boxplot(x=coluna, ax=ax1)
    ax2.set_xlim(limites(coluna))
    sns.boxplot(x=coluna, ax=ax2)


# In[19]:


diagrama_caixa(base_population['Year'])


# ### Modelo de Previsao
# 
# - Metricas de avaliacao

# In[20]:


def avaliar_modelo(nome_modelo, y_test, previsao):
    r2 = r2_score(y_test, previsao)
    RSME = np.sqrt(mean_squared_error(y_test, previsao))
    return f'Modelo {nome_modelo}:\R2:{r2:.2%}\RSME{RSME:.2f}'


# - Modelos a serem testados:
#  1. Radom Forest
#  2. Linear Regressionm
#  3. Extra Tree

# In[21]:


modelos = {'RadomForest': RandomForestRegressor(),
          'LinearRegression': LinearRegression(),
          'ExtraTree': ExtraTreesRegressor()}

y = base_population['Fertility Rate']
x = base_population.drop('Fertility Rate', axis=1)


# - Separar os dados em treino e teste + Treino do Modelo

# In[22]:


x_train, x_test, y_train, y_test = train_test_split(x,y,random_state = 10)

for nome_modelo, modelo in modelos.items():
    #Treinar
    modelo.fit(x_train, y_train)
    #Testar
    previsao = modelo.predict(x_test)
    print(avaliar_modelo(nome_modelo, y_test, previsao))


# Modelo Escolhido como melhor modelo: ExtraTressRegressor.
# Esse foi o modelo que apresentou o maior valor de R2 e ao mesmo tempo o menor valor do RSME. Como nao tivemos uma grande diferenca de velocidadede treino e de previsao desse modelo com o modelo de RandomForest (que teve os ressultados proximos de R2 E RSME, vamos escolher o modelo ExtraTrees).
# 
# O modelo de reressao linear obteve um resultado satisfatorio, porem os valores de R2 e RSME foram inferiores ao outros dois modelos.
# 
# Resultados das Metricas de avalicao no Modelo Vencedor:
# Modelo ExtraTree:
# R2:95.66%
# RSME0.41

# ### Ajustes e Melhorias no Melhor Modelo

# In[23]:


print(modelos['RadomForest'].feature_importances_)
print(x_train.columns)

importancia_features = pd.DataFrame(modelos['RadomForest'].feature_importances_, x_train.columns)
importancia_features = importancia_features.sort_values(by=0, ascending=False)
display(importancia_features)


# In[24]:


plt.figure(figsize=(15,5))
ax = sns.barplot(x=importancia_features.index, y=importancia_features[0])  
ax.tick_params(axis='x', rotation=90)

