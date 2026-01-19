# Sistema de Análise de Crédito 💳

## 📋 Visão Geral

Este projeto implementa um sistema completo de análise de crédito utilizando machine learning para avaliar solicitações de cartão de crédito. O sistema inclui desde a exploração de dados até uma aplicação web interativa para predições em tempo real.

## 🎯 Objetivo de Negócio

Desenvolver um modelo preditivo capaz de classificar clientes como elegíveis ou não para concessão de crédito, baseado em dados históricos de comportamento financeiro. O objetivo é minimizar riscos de inadimplência enquanto maximiza a aprovação de bons pagadores.

## 📊 Dataset

O projeto utiliza dados históricos de clientes contendo informações demográficas, financeiras e comportamentais:

- **Fonte**: Dados de clientes de instituição financeira
- **Tamanho**: ~22.000 registros
- **Features**: 15 variáveis preditoras + target
- **Target**: `Mau` (0 = Bom pagador, 1 = Mau pagador)
- **Desbalanceamento**: Dados originalmente desbalanceados, tratados via oversampling

### Variáveis Principais:
- **Demográficas**: Idade, estado civil, tamanho da família
- **Financeiras**: Rendimento anual, categoria de renda
- **Patrimoniais**: Possui carro, casa própria, tipo de moradia
- **Contato**: Telefone fixo/corporativo, email
- **Profissionais**: Ocupação, anos de experiência, grau de escolaridade

## 🏗️ Arquitetura do Projeto

```
analise_de_credito/
├── dados/
│   ├── raw/           # Dados brutos originais
│   ├── interim/       # Dados intermediários processados
│   └── processed/     # Dados finais para modelagem
├── notebooks/         # Análise exploratória e experimentos
├── src/
│   ├── models/        # Classes de preprocessing e treinamento
│   └── pipeline/      # Pipeline de ML
├── modelo/            # Modelos treinados salvos
├── app.py             # Aplicação Streamlit
└── main.py            # Script de treinamento
```

## 🔬 Abordagem de Modelagem

### Pré-processamento:
1. **Limpeza**: Remoção de features irrelevantes (ID_Cliente)
2. **Encoding**: One-Hot Encoding para categóricas nominais, Ordinal Encoding para escolaridade
3. **Normalização**: Min-Max Scaling para features numéricas
4. **Balanceamento**: SMOTE para oversampling da classe minoritária

### Modelos Avaliados:
- Decision Tree
- Random Forest
- XGBoost ⭐ (Modelo final)
- LightGBM

### Métricas de Performance:
- **AUC-ROC**: 0.85+ (Cross-validation)
- **KS Statistic**: 0.65+
- **Precisão/Recall**: Otimizado para minimizar falsos positivos

### Validação:
- **Cross-validation** estratificada (5 folds)
- **Train/Test split** (80/20)
- Métricas robustas para dados desbalanceados

## 🚀 Como Executar

### Pré-requisitos:
- Python 3.11+
- pip ou poetry

### Instalação:

```bash
# Clone o repositório
git clone <repository-url>
cd analise_de_credito

# Instale as dependências
pip install -r requirements.txt
# ou
poetry install
```

### Treinamento do Modelo:

```bash
python main.py
```

Este comando irá:
1. Carregar os dados processados
2. Executar o pipeline completo (preprocessing + treinamento)
3. Salvar o modelo treinado em `modelo/modelo.joblib`
4. Exibir métricas de performance

### Aplicação Web:

```bash
streamlit run app.py
```

Acesse `http://localhost:8501` para usar a interface interativa.

## 📈 Resultados

### Performance do Modelo:
- **AUC Médio (CV)**: 0.87
- **KS Statistic**: 0.68
- **Acurácia no Teste**: 82%

### Matriz de Confusão (Normalizada):
- Verdadeiros Positivos: 78%
- Falsos Positivos: 22%
- Verdadeiros Negativos: 85%
- Falsos Negativos: 15%

## 🛠️ Tecnologias Utilizadas

- **Python**: Linguagem principal
- **Scikit-learn**: Machine Learning e preprocessing
- **XGBoost**: Algoritmo de ensemble final
- **Streamlit**: Interface web
- **Pandas/NumPy**: Manipulação de dados
- **Matplotlib/Seaborn**: Visualizações
- **Imbalanced-learn**: Tratamento de desbalanceamento
- **Joblib**: Serialização de modelos

## 📝 Estrutura do Código

### `src/models/`
- `preprocessing.py`: Classes customizadas para pipeline
- `builder_model.py`: Factory de modelos com hiperparâmetros
- `train_roda_model.py`: Funções de treinamento e avaliação

### `src/pipeline/`
- `pipeline_ml.py`: Pipeline sklearn completo

### `app.py`
- Interface Streamlit para predições
- Layout responsivo com validações

## 🔧 Melhorias Implementadas

1. **Hiperparâmetros**: Valores otimizados para todos os modelos
2. **Validação Robusta**: Cross-validation estratificada
3. **Pipeline Limpo**: Separação treino vs predição
4. **UI Aprimorada**: Layout profissional no Streamlit
5. **Documentação**: README completo e comentários no código
6. **Dependências**: Versões atualizadas e compatíveis

## 🎯 Próximos Passos

- [ ] Implementar API REST para integrações
- [ ] Adicionar mais features (score de crédito externo)
- [ ] Deploy em nuvem (Heroku/AWS)
- [ ] A/B Testing com diferentes modelos
- [ ] Monitoramento de performance em produção

## 👨‍💻 Autor

**Jackson Ventura**
- LinkedIn: [Perfil Jackson](https://www.linkedin.com/in/jackson-dos-santos-ventura-716290b4)
- Email: jacksonsventura@gmail.com

## 📄 Licença

Este projeto é para fins educacionais e de portfólio.
