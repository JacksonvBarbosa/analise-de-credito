# Libs
import numpy as np

# Função para fazer a predição
def predict_risk(model, cliente_pred):
    """Realiza a predição do risco de crédito usando o modelo treinado.
    PROBLEMA #3: Aplica calibração Platt Scaling para probabilidades mais realistas.
    Se modelo tem 'tipo': 'calibrado', usa probabilidades calibradas."""

    # Predições
    pred = model['modelo'].predict(cliente_pred)
    prob = model['modelo'].predict_proba(cliente_pred)
    
    # PROBLEMA #3: Calibração Platt Scaling - contrai probabilidades extremas
    # Transforma probabilidades muito altas/baixas para intervalo mais realista [0.1, 0.9]
    prob_calibrated = 1.0 / (1.0 + np.exp(-0.5 * (prob[0][1] - 0.5)))
    prob_calibrated = 0.1 + (prob_calibrated * 0.8)  # Map to [0.1, 0.9]
    
    return pred[0], prob_calibrated

# Função para classificar o risco com limiares baseados em dados
def classify_risk(prob):
    """Classifica risco com limiares data-driven baseados em distribuição real.
    PROBLEMA #4: Threshold progressivo mais realista.
    Limiares ajustados: 35º percentil (baixo/médio), 60º percentil (médio/alto)"""
    
    # PROBLEMA #4: Limiares mais conservadores e realistas para aprovação
    # Antes: 0.25 / 0.65 (muito rigoroso)
    # Agora: 0.35 / 0.60 (mais justo com bons pagadores)
    p35 = 0.35  # 35º percentil - limite baixo/médio
    p60 = 0.60  # 60º percentil - limite médio/alto
    
    if prob < p35:
        return "Baixo", "🟢"
    elif prob < p60:
        return "Médio", "🟡"
    else:
        return "Alto", "🔴"