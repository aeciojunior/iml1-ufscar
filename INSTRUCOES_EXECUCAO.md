# Instruções de Execução - Projeto de Previsão de Leads

## 📋 Pré-requisitos

Antes de executar o projeto, certifique-se de ter:

1. **Python 3.8+** instalado no seu sistema
2. **pip** (gerenciador de pacotes Python)
3. **Git** (opcional, para clonar o repositório)

## 🚀 Instalação

### 1. Clone ou baixe o projeto
```bash
# Se usar Git:
git clone https://github.com/aeciojunior/iml1-ufscar.git
cd iml1-ufscar

# Ou simplesmente extraia o arquivo ZIP na pasta desejada
```

### 2. Instale as dependências
```bash
pip install -r requirements.txt
```

**Nota:** Se você estiver usando um ambiente virtual (recomendado), ative-o primeiro:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate

# Depois instale as dependências
pip install -r requirements.txt
```

## 📊 Execução da Análise Completa

### Executar o script Python principal
```bash
python analise_leads_completa.py
```

## 🔍 O que cada arquivo faz

### `analise_leads_completa.py`
- **Arquivo principal** com toda a análise
- Carrega e analisa os dados
- Cria features derivadas
- Testa diferentes modelos de regressão
- Otimiza o melhor modelo
- Faz previsões para novos dados

### `exemplo_uso.py`
- Demonstra como usar o modelo treinado
- Cria dados de exemplo
- Simula previsões
- Fornece recomendações

### `Unidade1 - Atividade1.csv`
- Base de dados com informações de investimentos e leads
- **NÃO MODIFIQUE** este arquivo

## 📈 Fluxo de Execução

1. **Carregamento dos Dados**
   - Leitura do arquivo CSV
   - Verificação de qualidade dos dados

2. **Análise Exploratória**
   - Estatísticas descritivas
   - Visualizações
   - Análise da variável target (leads)

3. **Engenharia de Features**
   - Criação de variáveis agregadas
   - Cálculo de proporções
   - Features de diversificação

4. **Preparação dos Dados**
   - Divisão treino/teste
   - Normalização
   - Limpeza de dados

5. **Teste de Modelos**
   - Regressão Linear
   - Regressão Ridge/Lasso
   - Random Forest
   - XGBoost
   - SVR

6. **Otimização**
   - Grid Search para o melhor modelo
   - Ajuste de hiperparâmetros

7. **Análise de Features**
   - Importância das variáveis
   - Visualizações

8. **Previsões**
   - Exemplo com novos dados
   - Análise de resultados

## ⚠️ Solução de Problemas

### Erro: "ModuleNotFoundError"
```bash
# Instale as dependências novamente
pip install -r requirements.txt
```

### Erro: "Permission denied"
```bash
# No Windows, execute o PowerShell como administrador
# No Linux/Mac, use sudo se necessário
```

### Erro: "MemoryError"
- Reduza o tamanho do dataset
- Use menos features
- Execute em um computador com mais RAM

### Erro: "FileNotFoundError"
- Verifique se o arquivo `Unidade1 - Atividade1.csv` está na pasta correta
- Verifique o caminho do arquivo

## 📊 Interpretação dos Resultados

### Métricas de Performance
- **R² Score**: Quanto mais próximo de 1, melhor
- **MAE**: Erro absoluto médio (menor é melhor)
- **RMSE**: Raiz do erro quadrático médio (menor é melhor)

### Features Mais Importantes
- As features com maior importância são as que mais influenciam a geração de leads
- Use essas informações para otimizar investimentos

### Previsões
- O modelo prevê a quantidade de leads baseado nos investimentos
- Use para planejar campanhas de marketing

## 🔧 Personalização

### Modificar Features
- Edite a função `criar_features()` em `analise_leads_completa.py`
- Adicione ou remova variáveis conforme necessário

### Testar Novos Modelos
- Adicione novos algoritmos na função `testar_modelos()`
- Ajuste hiperparâmetros na função `otimizar_melhor_modelo()`

### Novos Dados
- Substitua o arquivo CSV por seus próprios dados
- Mantenha a mesma estrutura de colunas

## 📝 Relatório da Atividade

Após executar a análise, você terá:

1. **Análise exploratória completa** dos dados
2. **Comparação de modelos** com métricas de performance
3. **Modelo otimizado** (XGBoost) para previsões
4. **Análise de importância** das features
5. **Sistema de previsão** funcionando
6. **Exemplos práticos** de uso

## 🎯 Próximos Passos

1. **Implementar em produção** para previsões em tempo real
2. **Criar dashboard** para monitoramento
3. **Coletar mais dados** para melhorar a performance
4. **Aplicar técnicas de otimização** para maximizar ROI

## 📞 Suporte

Se encontrar problemas:

1. Verifique se todas as dependências estão instaladas
2. Confirme que o arquivo CSV está na pasta correta
3. Verifique a versão do Python (3.8+)
4. Execute o script passo a passo para identificar onde está o erro

## ✅ Checklist de Execução

- [ ] Python 3.8+ instalado
- [ ] Dependências instaladas (`pip install -r requirements.txt`)
- [ ] Arquivo CSV na pasta correta
- [ ] Script executado com sucesso
- [ ] Resultados analisados
- [ ] Modelo funcionando para previsões
