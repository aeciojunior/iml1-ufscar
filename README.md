# IML1.1 - Previsão de Leads

## Descrição do Projeto
Este projeto implementa um sistema de previsão de leads baseado em dados de investimentos em marketing digital. O objetivo é encontrar o melhor modelo de regressão para prever a quantidade de leads gerados com base nos investimentos em diferentes canais de marketing.

## Estrutura do Projeto
- `notebook_analise_leads.ipynb`: Notebook principal com toda a análise e modelagem
- `Unidade1 - Atividade1.csv`: Base de dados com informações de investimentos e leads
- `README.md`: Este arquivo de documentação

## Variáveis do Dataset
- **Branding**: Campanhas que apresentam a ideologia, autenticidade, visão e personalidade da empresa
- **DPA**: Anúncios dinâmicos de produtos (promovem catálogos inteiros de produtos)
- **DRA**: Remarketing dinâmico (exibe anúncios personalizados para usuários que já visitaram o site)
- **Display**: Anúncios visualmente atraentes para um público relevante
- **Rmkt**: Campanhas de remarketing
- **Search**: Campanha publicitária que exibe anúncios de texto nas páginas de resultados de pesquisa
- **Shopping**: Campanha no Google Ads que exibe anúncios de produtos com informações detalhadas
- **Flag**: Indicadora de pandemia (0 = sem pandemia, 1 = com pandemia)
- **Leads**: Variável target (quantidade de interesse no produto de uma empresa)

## Metodologia
1. **Análise Exploratória dos Dados**: Compreensão da estrutura e distribuição dos dados
2. **Engenharia de Features**: Criação de novas variáveis e agregações
3. **Teste de Modelos**: Avaliação de diferentes algoritmos de regressão
4. **Seleção do Melhor Modelo**: Comparação baseada em métricas de performance
5. **Previsão**: Aplicação do modelo selecionado para novos dados

## Modelos Testados
- Regressão Linear
- Regressão Ridge
- Regressão Lasso
- Random Forest
- XGBoost
- Support Vector Regression (SVR)

## Critérios de Avaliação
- R² Score
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Cross-validation score

## Como Executar
1. Instalar as dependências: `pip install -r requirements.txt`
2. Abrir o notebook `notebook_analise_leads.ipynb`
3. Executar todas as células sequencialmente

## Autor
[Seu Nome] - IML1 - UFSCar 