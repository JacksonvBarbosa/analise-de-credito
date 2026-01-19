# Libs
import joblib

# Models
from sklearn.pipeline import Pipeline
from src.models.preprocessing import (DropFeatures, OneHotEncodingNames,
                                        OrdinalFeature, MinMax, MinMaxScaler,
                                        Oversample)
from src.models.train_roda_model import train_deploy_model


# Função Pipeline
import os
import joblib

def pipeline(df, modelo: str, save=False, nome: str = "modelo"):

    pipeline = Pipeline([
        ('feature_dropper', DropFeatures()),
        ('OneHotEncoding', OneHotEncodingNames()),
        ('ordinal_feature', OrdinalFeature()),
        ('min_max_scaler', MinMax()),
        ('oversample', Oversample())
    ])

    df_pipeline = pipeline.fit_transform(df)
    model = train_deploy_model(df_pipeline, modelo)

    if save:
        caminho_modelo = os.path.join("modelo")
        os.makedirs(caminho_modelo, exist_ok=True)

        caminho_arquivo = os.path.join(caminho_modelo, f"{nome}.joblib")
        joblib.dump(model, caminho_arquivo)

        print(f"Modelo salvo em: {caminho_arquivo}")
    else:
        print("Arquivo não foi salvo para utilização em produção")

    return model

def pipeline_teste(df):

    pipeline = Pipeline([
        ('feature_dropper', DropFeatures()),
        ('OneHotEncoding', OneHotEncodingNames()),
        ('ordinal_feature', OrdinalFeature()),
        ('min_max_scaler', MinMax()),
    ])
    df_pipeline = pipeline.fit_transform(df)
    return df_pipeline