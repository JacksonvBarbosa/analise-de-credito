# Libs
import pandas as pd

# models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def build_models(modelo: str):
    '''
    Construtor de modelos com hiperparâmetros otimizados para credit scoring.
    Parâmetros conservadores para reduzir overfitting e melhorar generalização.
    
    :param model: 'decisiontree', 'randomforest', 'xgboost', 'lgmboost'
    '''

    if modelo.lower() == 'decisiontree':
        model = DecisionTreeClassifier(max_depth=4, random_state=1561651)
    elif modelo.lower() == 'randomforest':
        model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=1561651)
    elif modelo.lower() == 'xgboost':
        # Hiperparâmetros conservadores para crédito
        model = XGBClassifier(
            max_depth=4,              # Reduzido de 6 (menos overfitting)
            learning_rate=0.05,       # Mais conservador (default=0.3)
            n_estimators=200,         # Mais árvores, cada uma simples
            subsample=0.8,            # Regularização
            colsample_bytree=0.8,     # Reduz features por árvore
            reg_alpha=0.1,            # L1 regularization
            reg_lambda=1.0,           # L2 regularization
            min_child_weight=3,       # Nós menores não dividem facilmente
            random_state=1561651,
            verbosity=0
        )
    elif modelo.lower() == 'lgmboost':
        model = LGBMClassifier(
            max_depth=4,
            learning_rate=0.05,
            n_estimators=200,
            num_leaves=15,
            random_state=1561651,
            verbose=-1
        )

    return model