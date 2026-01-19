#%%
# Libs
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# %%

df_cadastrados = pd.read_csv("../../dados/raw/clientes_cadastrados.csv")

# %%
# Observar dados duplicados
df_cadastrados.duplicated().sum()

# %%
# Mostrar quantidade de duplicados dentro do dataset
df_cadastrados[df_cadastrados['ID_Cliente'].duplicated(keep=False)].shape

# %%
# Excluir dados de IDs duplicados
id_duplicados = df_cadastrados[df_cadastrados['ID_Cliente'].duplicated(keep=False)]
df_cadastrados_limpo = df_cadastrados.drop(id_duplicados.index)

#%%
# Mostrar quantidade de dados que sobrou ao serem excluidos os dados duplicados
df_cadastrados_limpo['ID_Cliente'].value_counts()

#%%
# Observar dados nulos
df_cadastrados_limpo.isnull().sum()

#%%
# Na coluna Ocupação temos valores nulos 134.177 especificamente
df_cadastrados_limpo['Ocupacao'].unique()

#%%
# Preenchendo missing values com categoria explícita (não genérica)
# Melhor que "Outro": identifica clientes que não informaram ocupação
df_cadastrados_limpo['Ocupacao'] = df_cadastrados_limpo['Ocupacao'].fillna(value="Nao_informada")
# %%
# Conta valores únicos dentro de cada feature
df_cadastrados_limpo.nunique()

# %%
# Mapear e substituir valores categoricos por numéricos
dict_y_n = {"Y" : 1, "N" : 0}

df_cadastrados_limpo['Tem_carro'] = df_cadastrados_limpo['Tem_carro'].map(dict_y_n)
df_cadastrados_limpo['Tem_casa_propria'] = df_cadastrados_limpo['Tem_casa_propria'].map(dict_y_n)

#%%
# Mostrar valores modificados
df_cadastrados_limpo[['Tem_carro', 'Tem_casa_propria']]

#%%
# Mostra como estão o valores da feature Idade
df_cadastrados_limpo['Idade']
#%%
# Tratamento de feature Idade que estão sendo apresentadas em dias e o simbolo (-) representa os
# dias passados. Estou utilizando 365 dias e 2425 para média real ano bisexto e correções no
# calendário gregoriano média utilizada em estatística, demográfia, finanças e ciência de dados.
df_cadastrados_limpo['Idade'] = -df_cadastrados_limpo['Idade'] / 365.2425
df_cadastrados_limpo['Idade']

#%%
# Estou utilizando a mesma técnica de conversão de anos da feature Idade para Anos Empregado
df_cadastrados_limpo['Anos_empregado'] = -df_cadastrados_limpo['Anos_empregado'] / 365.2425
df_cadastrados_limpo['Anos_empregado'].value_counts()

#%%
# Ajustando os valores abaixo de 0, para que para que fiquem zerado assim sera considerado que
# o cliente está sem ocupação
df_cadastrados_limpo.loc[df_cadastrados_limpo['Anos_empregado']<0, 'Anos_empregado']=0
df_cadastrados_limpo['Anos_empregado'].value_counts()

#%%
# Em nosso dataset todos os clientes tem celular.
df_cadastrados_limpo.loc[df_cadastrados_limpo['Tem_celular']==1]
# %%
# Dropando as features Genero e Tem_celular, 1º não vou avaliar credito do cliente por genero,
# 2º todos os clientes tem celular então é um feature que não é necessária para nosso modelo.
df_cadastrados_limpo.drop(columns=["Genero", "Tem_celular"], inplace=True)

# %%
# Averigua alta correlação que pode influência na hora da utilização do modelo de predição
corr = df_cadastrados_limpo.corr(numeric_only=True)
plt.figure(figsize=(20, 10))
sns.heatmap(corr, cmap="Blues", annot=True)

#%
# Com uma correlação quase de 90% positiva, irei dropar a feature Qtd_filhos para evitar um
# Overfiting do modelo de predição
df_cadastrados_limpo.drop("Qtd_filhos", axis=1, inplace=True)

#%%
# Reorganizando as colunas do dataset
df_cadastrados_limpo=df_cadastrados_limpo[['ID_Cliente', 'Tem_carro', 'Tem_casa_propria', 
                                           'Tem_telefone_trabalho', 'Tem_telefone_fixo', 'Tem_email',
                                           'Tamanho_familia', 'Rendimento_anual', 'Idade', 'Anos_empregado',
                                           'Categoria_de_renda', 'Grau_escolaridade', 'Estado_civil',
                                           'Moradia', 'Ocupacao']]

#%%

'''Winsorização de extremos em vez de remoção (mantém dados, limita valores extremos)'''

coluna = df_cadastrados_limpo['Rendimento_anual']

coluna_med = coluna.mean()
coluna_std = coluna.std()

# Definir limites para winsorização (1º e 99º percentil = mais seguro que ±2σ)
from scipy.stats import mstats
df_cadastrados_limpo['Rendimento_anual'] = mstats.winsorize(
    df_cadastrados_limpo['Rendimento_anual'], 
    limits=[0.01, 0.01]  # Limita top/bottom 1% ao invés de remover
)

# Sem remoção de linhas: preserva todos os clientes
df_clientes_cadastrados_sem_outliers = df_cadastrados_limpo
df_clientes_cadastrados_sem_outliers.shape

#%%

df_clientes_cadastrados_tratamento1 = df_clientes_cadastrados_sem_outliers
df_clientes_cadastrados_tratamento1.shape
#%%
# Salva Dados tratados dos clientes para análise dos dados prévia antes de continuar
# o restante dos tratamentos.
df_cadastrados_limpo.to_csv("../../dados/interim/df_analises.csv", index=False)

#%%
# Tratando dados do dataset dos clientes_aprovados.csv
df_aprovados = pd.read_csv('../../dados/raw/clientes_aprovados.csv')
df_aprovados.head()

#%%
# Obsevando infromações básicas do dataset
df_aprovados.info()

#%%
# Observando um dos clientes
df_aprovados.query('ID_Cliente == 5001712')

#%%
# Análise de outro cliente
df_aprovados.query('ID_Cliente == 5001711')

#%%
# Observando o atributo de faixa de atrasado
df_aprovados.Faixa_atraso.value_counts().index.to_list()

#%%
# Agrupando os registros dos creditos e averiguando quantos dias foi aprovado a liberação do crédito
df_registros_credi_agrup_ID = df_aprovados.groupby('ID_Cliente')
dia_abertura = df_registros_credi_agrup_ID.apply(lambda x: min(x['Mes_referencia']))
dia_abertura.name = 'Abertura'
dia_abertura

#%%
# Cria um Dataframe fazendo um merge no dataframe df_aprovados com os valores do dia_abertura indexando pelo ID_Cliente
df_clientes_aprovados = df_aprovados.merge(dia_abertura, on="ID_Cliente")
df_clientes_aprovados

#%%
# Agrupando os registros dos cr´ditos e averiguando quantos dias acima da data limite os cliente estão inadimplentes
dia_final = df_registros_credi_agrup_ID.apply(lambda x: max(x['Mes_referencia']))
dia_final.name = 'Final'
dia_final

#%%
# Inserindo valor de clientes inadimplentes no dataframe df_clientes_aprovados
df_clientes_aprovados = df_clientes_aprovados.merge(dia_final, on='ID_Cliente')
df_clientes_aprovados

#%%
# Cria janela de abertura e prazo final de crádito
df_clientes_aprovados['Janela'] = df_clientes_aprovados['Final'] - df_clientes_aprovados['Abertura']
df_clientes_aprovados

#%%
# Cria atributo MOB (Months on Book) representa o número de meses desde a abaertura do contrato, conta ou crédito do cliente
df_clientes_aprovados['MOB'] = df_clientes_aprovados['Mes_referencia'] - df_clientes_aprovados['Abertura']
df_clientes_aprovados

#%%
# Cria um dicionário para transformar os dados categoricos em númericos assim nosso modelo irá conseguir 
# trabalhar corretamente.
dict_faixa_atraso_ind ={'nenhum empréstimo': 0, 'pagamento realizado': 1,
                        '1-29 dias': 2, '30-59 dias': 3, '60-89 dias': 4,
                        '90-119 dias': 5, '120-149 dias': 6, '>150 dias': 7}
#%%
# Cria um atributo númerico para que o nosso ML consiga trabalhar corretamente
df_clientes_aprovados['Ind_Faixa_atraso'] = df_clientes_aprovados['Faixa_atraso'].map(dict_faixa_atraso_ind)
df_clientes_aprovados

#%%
# Cria atributo mau pagador, mau pagador para cliente com falta de pagamento
# acima de 59 dias
df_clientes_aprovados['Mau'] = df_clientes_aprovados.apply(lambda x: 1 if x['Ind_Faixa_atraso'] > 3 else 0, axis=1)
df_clientes_aprovados

#%%
# Cria dataframe de registro de créditos agrupando pelo ID_Cliente e utilizando as colunas ID_CLiente, Abertura, Final e Janela
df_registros_creditos_ID = df_clientes_aprovados[['ID_Cliente','Abertura', 'Final', 'Janela']].groupby('ID_Cliente').apply(lambda x: x.iloc[0]).reset_index(drop=True)
df_registros_creditos_ID

#%%
# Obeservar quantos clientes abriram crédito em cada período de abertura.
df_denominador = (
    df_registros_creditos_ID
    .groupby('Abertura')['ID_Cliente']
    .count()
    .reset_index(name='Qtd_Clientes')
)
df_denominador

#%%
# Cria dataframe de registro de créditos agrupando pelo ID_Cliente e utilizando as colunas ID_CLiente, Abertura, Final e Janela
df_vintage = df_clientes_aprovados.groupby(['Abertura','MOB']).apply(lambda X: X['ID_Cliente'].count()).reset_index()
df_vintage.columns = ['Abertura','MOB','Qtd_Clientes']
df_vintage

#%%
# Efetuar o merge do vintage com o denominador utilizando a abertura como referência
df_vintage = pd.merge(df_vintage[['Abertura','MOB']], df_denominador, on = ['Abertura'], how = 'left')
df_vintage

#%%
# 1. Filtra apenas clientes maus
df_mau = df_clientes_aprovados[df_clientes_aprovados['Mau'] == 1]

# 2. Mantém apenas o primeiro MOB em que o cliente virou mau por safra
df_mau_first = (
    df_mau
    .sort_values('MOB')
    .drop_duplicates(subset=['Abertura', 'ID_Cliente'], keep='first')
)

# 3. Conta clientes maus por Abertura x MOB
mau_por_mob = (
    df_mau_first
    .groupby(['Abertura', 'MOB'])['ID_Cliente']
    .nunique()
    .reset_index(name='Qtd_Mau')
)

# 4. Acumulado por MOB (curva de vintage)
mau_por_mob['Qtd_Mau'] = (
    mau_por_mob
    .sort_values('MOB')
    .groupby('Abertura')['Qtd_Mau']
    .cumsum()
)

# 5. Merge com o df_vintage
df_vintage = df_vintage.merge(
    mau_por_mob,
    on=['Abertura', 'MOB'],
    how='left'
)

# 6. Preenche NaN e calcula a taxa
df_vintage['Qtd_Mau'] = df_vintage['Qtd_Mau'].fillna(0)

df_vintage['Taxa_de_Mau'] = (
    df_vintage['Qtd_Mau'] / df_vintage['Qtd_Clientes']
)

df_vintage    

#%%
# Transforma dataset longo em largo
df_vintage_pivot = df_vintage.pivot(index = 'Abertura',
                             columns = 'MOB',
                             values = 'Taxa_de_Mau')

df_vintage_pivot

#%%
# Dicionário de intervalo de atraso
dict_intervalo_atraso = {'maior_30_dias': 3, 'maior_60_dias': 4, 'maior_90_dias': 5,
                        'maior_120_dias': 6, 'maior_150_dias': 7}

#%%
# 
for chave, valor in dict_intervalo_atraso.items():
  df_clientes_aprovados[f'Mau_{chave}'] = df_clientes_aprovados.apply(lambda x: 1 if x['Ind_Faixa_atraso'] >= valor else 0, axis=1) # mais de 60
df_clientes_aprovados.head()
# %%
# Cria dataframe de taxa de mau pagador
dict_taxa_mau = {}
id_sum = len(set(df_clientes_aprovados['ID_Cliente']))

for chave in dict_intervalo_atraso.keys():
  df_min_mau = df_clientes_aprovados.query(f'Mau_{chave} == 1').groupby('ID_Cliente')['MOB'].min().reset_index()
  df_mob_taxa_mau = pd.DataFrame({'MOB':range(0,61), 'Taxa_Mau': np.nan})
  lst = []
  for i in range(0,61):
      due = df_min_mau.query('MOB == @i')['ID_Cliente'].to_list()
      lst.extend(due) #cumsum
      df_mob_taxa_mau.loc[df_mob_taxa_mau['MOB'] == i, 'Taxa_Mau'] = len(set(lst)) / id_sum
  dict_taxa_mau[chave] = df_mob_taxa_mau['Taxa_Mau']

df_taxa_mau = pd.DataFrame(dict_taxa_mau)
df_taxa_mau

#%%
# Plota gráfico com clientes maus pagadores com faixas de atraso
df_taxa_mau.plot(grid = True, title = '% acumulado de clientes maus para diversas faixas de atraso', figsize=(10, 6))
plt.xlabel('MOB')
plt.ylabel('% acumulado de clientes')
plt.show()

#%%

len(df_clientes_aprovados.query('Janela >= 12').groupby('ID_Cliente').count().index)

#%%

df_clientes_aprovados_tratamento1 = df_clientes_aprovados.query('Janela >= 12').copy()
df_clientes_aprovados_tratamento1.shape
#%%
# Função para verificar registros
def verifica(registros):
  lista_status = registros['Faixa_atraso'].to_list()
  if '60-89 dias' in lista_status or '90-119 dias' in lista_status or '120-149 dias' in lista_status or '>150 dias' in lista_status:
    return 1
  else:
    return 0

# Cria Dataframe de registros de credito  
df_registros_creditos_id = pd.DataFrame(df_clientes_aprovados_tratamento1.groupby('ID_Cliente').apply(verifica)).reset_index()
df_registros_creditos_id.columns = ['ID_Cliente', 'Mau']
df_registros_creditos_id.head()

#%%

df_registro_clientes_targets = df_clientes_cadastrados_tratamento1.merge(df_registros_creditos_id, on='ID_Cliente')
df_registro_clientes_targets.head()

#%%
# Taxa de bons e maus pagadores
df_registro_clientes_targets['Mau'].value_counts(normalize=True)*100

#%%
# Salva Dataframe pronto para análise e utilização em modelos
df_registro_clientes_targets.to_csv("..\..\dados\processed\df_analises_models.csv", index=False)
