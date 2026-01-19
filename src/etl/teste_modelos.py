#%%
# Libs
import pandas as pd

# Modelos
from src.pipeline.pipeline_ml import pipeline
from main import main

#%%
main()

#%%
# carregar DataFrame
df = pd.read_csv('..\..\dados\processed\df_analises_models.csv') 
#%%
modelo = pipeline(df, 'xgboost', save=False)

#%%
print(modelo['prob_predic'][0])
print(modelo['predicao'][0])
# %%
import joblib
from src.pipeline.pipeline_ml import pipeline_teste
from src.models.train_roda_model import data_split

#%%
# Lista de todas as variáveis: 
novo_cliente = [#0, # ID_Cliente
                    1, # Tem_carro
                    0, # Tem_Casa_Propria
                    1, # Tem_telefone_trabalho
                    1, # Tem_telefone_fixo
                    0,  # Tem_email
                    2,  # Tamanho_Familia
                    315000.0, # Rendimento_anual	
                    37,#37.1178052937432, # Idade
                    2,#1.6044135060952651, # Anos_empregado
                    'Associado comercial', # Categoria_de_renda
                    'Ensino superior', # Grau_Escolaridade
                    'Casado', # Estado_Civil	
                    'Casa/apartamento próprio', # Moradia                                                  
                    'Outro', # Ocupacao
                     0 # target (Mau)
                    ]

#%%
# Separando os dados em treino e teste
treino_df, teste_df = data_split(df, 0.2)

#%%
print(teste_df.columns)
print(len(df.columns))
print(len(novo_cliente))
#%%
#Criando novo cliente
cliente_predict_df = pd.DataFrame([novo_cliente],columns=teste_df.columns)

#%%
#Concatenando novo cliente ao dataframe dos dados de teste
teste_novo_cliente  = pd.concat([teste_df,cliente_predict_df],ignore_index=True)

#%%
#Aplicando a pipeline
teste_novo_cliente = pipeline_teste(teste_novo_cliente)

#%%
#retirando a coluna target
cliente_pred = teste_novo_cliente.drop(['Mau'], axis=1)

#%%
model = joblib.load('../../modelo/modelo.joblib')
final_pred = model['modelo'].predict(cliente_pred)
if final_pred[-1] == 0:
    print(final_pred[-1])
    print('### Parabéns! Você teve o cartão de crédito aprovado')
else:
    print(final_pred[-1])
    print('### Infelizmente, não podemos liberar crédito para você agora!')

#%%

df