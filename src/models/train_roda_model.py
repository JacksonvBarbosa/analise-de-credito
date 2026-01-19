#%%
# Libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Modelos
from sklearn import metrics
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from imblearn.over_sampling import SMOTE
from src.models.builder_model import build_models

# scipy
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

#|-------------------------Separação--------------------------|

# Cria df para o treino do modelo
# df = pd.read_csv("..\..\dados\processed\df_analises_models.csv")
# df.head()

# Semente para melhor reprodutibiliade
SEED = 1561651

# Função de train e test
def separate_train_test(df):
    X = df.drop('Mau', axis=1)
    y = df['Mau']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    return X_train, X_test, y_train, y_test

# Caso queira salvar df para backup individual
# def salva_df_train_test(train_df, test_df, nome_train = "train", nome_test = "test"):
#     df_train = nome_train
#     df_test = nome_test
#     train_df.to_csv(rf"..\..\dados\backup\{df_train}.csv", index=False)
#     test_df.to_csv(rf"..\..\dados\backup\{df_test}.csv", index=False)

# Separando os dados em treino e teste
def data_split(df, test_size):
    SEED = 1561651
    treino_df, teste_df = train_test_split(df, test_size=test_size, random_state=SEED)
    return treino_df.reset_index(drop=True), teste_df.reset_index(drop=True)

# Roda e treina modelo
def train_deploy_model(df, modelo: str):

    # Variveis de treino e teste
    X_train, X_test, y_train, y_test = separate_train_test(df)

    # definir modelo ('decisiontree', 'randomforest', 'xgboost', 'lgmboost')
    modelo = build_models(modelo)

    # Treinando modelo com os dados de treino
    modelo_treinado = modelo.fit(X_train, y_train)

    # Calculando a probabilidade e calculando o AUC
    prob_predic = modelo_treinado.predict_proba(X_test)

    print(f"\n------------------------------Resultados {modelo}------------------------------\n")

    auc = roc_auc_score(y_test, prob_predic[:,1])
    print(f"AUC {auc}")
    
    # Separando a probabilidade de ser bom e mau, e calculando o KS
    #métrica KS: probabilidade de um cliente ser classificado como bom ou mau. 
    data_bom = np.sort(modelo_treinado.predict_proba(X_test)[:, 0])
    data_mau = np.sort(modelo_treinado.predict_proba(X_test)[:, 1])
    kstest = stats.ks_2samp(data_bom, data_mau)

    print(f"Métrica KS: {kstest}")

    print("\nConfusion Matrix\n")
    # Criando matriz de confusão
    fig, ax = plt.subplots(figsize=(7,7))
    matriz_confusao = ConfusionMatrixDisplay.from_estimator(modelo_treinado, X_test, y_test, normalize='true',
                                            display_labels=['Bom pagador', 'Mau pagador'],
                                            ax=ax, cmap=plt.cm.Blues)
    ax.set_title("Matriz de Confusão\n Normalizada", fontsize=16, fontweight="bold")
    ax.set_xlabel("Label predita", fontsize=18)
    ax.set_ylabel("Label verdadeira", fontsize=18)
    plt.grid(False)    
    plt.show()

    # Fazendo a predicao dos dados de teste e calculando o classification report
    predicao = modelo_treinado.predict(X_test)
    print("\nClassification Report")
    print(classification_report(y_test, predicao, zero_division=0))


    print("\nRoc Curve\n")
    metrics.RocCurveDisplay.from_estimator(modelo_treinado, X_test, y_test)
    plt.show()
    
    dict_model_metrics = {
        'modelo' : modelo_treinado,
        'prob_predic': prob_predic,
        'auc' : auc,
        'kstest' : kstest,
        'predicao' : predicao

    }

    return dict_model_metrics
