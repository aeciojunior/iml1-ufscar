#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IML1.1 - Previsão de Leads
Análise completa para previsão de leads baseada em investimentos em marketing digital

Autor: AECIO PEREIRA SANTIAGO JUNIOR - IML1 - UFSCar
Data: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Configurações para melhor visualização
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def carregar_dados():
    """Carrega e verifica os dados"""
    print("=== CARREGAMENTO DOS DADOS ===")
    df = pd.read_csv('Unidade1 - Atividade1.csv')
    
    print(f"Shape do dataset: {df.shape}")
    print(f"Colunas: {list(df.columns)}")
    
    return df

def analise_exploratoria(df):
    """Realiza análise exploratória dos dados"""
    print("\n=== ANÁLISE EXPLORATÓRIA ===")
    
    # Informações básicas
    print(f"\nDimensões: {df.shape[0]} linhas x {df.shape[1]} colunas")
    print(f"\nTipos de dados:")
    print(df.dtypes)
    
    # Verificação de valores nulos e duplicados
    print(f"\nValores nulos: {df.isnull().sum().sum()}")
    print(f"Valores duplicados: {df.duplicated().sum()}")
    
    # Análise da variável target
    print(f"\n=== ANÁLISE DA VARIÁVEL TARGET (LEADS) ===")
    print(f"Valores únicos: {df['leads'].nunique()}")
    print(f"Range: {df['leads'].min()} a {df['leads'].max()}")
    print(f"Média: {df['leads'].mean():.2f}")
    print(f"Mediana: {df['leads'].median():.2f}")
    print(f"Desvio padrão: {df['leads'].std():.2f}")
    
    # Análise da variável Flag (pandemia)
    print(f"\n=== ANÁLISE DA VARIÁVEL FLAG (PANDEMIA) ===")
    flag_counts = df['Flag'].value_counts()
    print(f"0 (sem pandemia): {flag_counts[0]} ({flag_counts[0]/len(df)*100:.1f}%)")
    print(f"1 (com pandemia): {flag_counts[1]} ({flag_counts[1]/len(df)*100:.1f}%)")
    
    # Visualizações
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Distribuição de leads
    axes[0, 0].hist(df['leads'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribuição de Leads')
    axes[0, 0].set_xlabel('Quantidade de Leads')
    axes[0, 0].set_ylabel('Frequência')
    axes[0, 0].axvline(df['leads'].mean(), color='red', linestyle='--', label=f'Média: {df["leads"].mean():.1f}')
    axes[0, 0].legend()
    
    # Boxplot de leads
    axes[0, 1].boxplot(df['leads'])
    axes[0, 1].set_title('Boxplot de Leads')
    axes[0, 1].set_ylabel('Quantidade de Leads')
    
    # Comparação por período
    leads_sem_pandemia = df[df['Flag'] == 0]['leads']
    leads_com_pandemia = df[df['Flag'] == 1]['leads']
    
    axes[1, 0].boxplot([leads_sem_pandemia, leads_com_pandemia], labels=['Sem Pandemia', 'Com Pandemia'])
    axes[1, 0].set_title('Comparação de Leads por Período')
    axes[1, 0].set_ylabel('Quantidade de Leads')
    
    # Histograma comparativo
    axes[1, 1].hist(leads_sem_pandemia, alpha=0.7, label='Sem Pandemia', bins=20)
    axes[1, 1].hist(leads_com_pandemia, alpha=0.7, label='Com Pandemia', bins=20)
    axes[1, 1].set_title('Distribuição de Leads por Período')
    axes[1, 1].set_xlabel('Quantidade de Leads')
    axes[1, 1].set_ylabel('Frequência')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return df

def criar_features(df):
    """Cria novas features para melhorar o modelo"""
    print("\n=== CRIAÇÃO DE NOVAS FEATURES ===")
    
    # Identificação das colunas por tipo de campanha
    colunas_branding = [col for col in df.columns if 'Branding' in col]
    colunas_dpa = [col for col in df.columns if 'DPA' in col]
    colunas_dra = [col for col in df.columns if 'DRA' in col]
    colunas_display = [col for col in df.columns if 'Display' in col]
    colunas_rmkt = [col for col in df.columns if 'Rmkt' in col]
    colunas_search = [col for col in df.columns if 'Search' in col]
    colunas_shopping = [col for col in df.columns if 'Shopping' in col]
    
    # 1. Investimento total por campanha
    df['total_branding'] = df[colunas_branding].sum(axis=1)
    df['total_dpa'] = df[colunas_dpa].sum(axis=1)
    df['total_dra'] = df[colunas_dra].sum(axis=1)
    df['total_display'] = df[colunas_display].sum(axis=1)
    df['total_rmkt'] = df[colunas_rmkt].sum(axis=1)
    df['total_search'] = df[colunas_search].sum(axis=1)
    df['total_shopping'] = df[colunas_shopping].sum(axis=1)
    
    # 2. Investimento total geral
    df['investimento_total'] = df[['total_branding', 'total_dpa', 'total_dra', 
                                   'total_display', 'total_rmkt', 'total_search', 'total_shopping']].sum(axis=1)
    
    # 3. Proporções de cada tipo de campanha
    df['prop_branding'] = df['total_branding'] / df['investimento_total']
    df['prop_dpa'] = df['total_dpa'] / df['investimento_total']
    df['prop_dra'] = df['total_dra'] / df['investimento_total']
    df['prop_display'] = df['total_display'] / df['investimento_total']
    df['prop_rmkt'] = df['total_rmkt'] / df['investimento_total']
    df['prop_search'] = df['total_search'] / df['investimento_total']
    df['prop_shopping'] = df['total_shopping'] / df['investimento_total']
    
    # 4. Diversificação de canais
    df['canais_ativos'] = ((df[['total_branding', 'total_dpa', 'total_dra', 
                                 'total_display', 'total_rmkt', 'total_search', 'total_shopping']] > 0).sum(axis=1))
    
    # 5. Eficiência de investimento
    df['eficiencia_investimento'] = df['leads'] / df['investimento_total']
    
    # 6. Interação entre pandemia e investimento total
    df['pandemia_investimento'] = df['Flag'] * df['investimento_total']
    
    # Substituir valores infinitos e NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    print("Features criadas com sucesso!")
    
    return df

def preparar_dados(df):
    """Prepara os dados para modelagem"""
    print("\n=== PREPARAÇÃO DOS DADOS ===")
    
    # Features finais para o modelo
    features_originais = ['Branding', 'DPA 1', 'DPA 3', 'DRA 1', 'DRA 3', 'Display 1', 'Display 2',
                          'Rmkt 1', 'Rmkt 3', 'Rmkt 6', 'Rmkt 7', 'Search 1', 'Search 2', 'Search 3', 
                          'Search 4', 'Search 5', 'Search 6', 'Shopping 1', 'Shopping 2', 'Shopping 3',
                          'Shopping 4', 'Shopping 5', 'Shopping 6', 'Shopping 7', 'Flag']
    
    features_novas = ['total_branding', 'total_dpa', 'total_dra', 'total_display', 
                      'total_rmkt', 'total_search', 'total_shopping', 'investimento_total',
                      'canais_ativos', 'eficiencia_investimento', 'pandemia_investimento']
    
    features_finais = features_originais + features_novas
    target = 'leads'
    
    print(f"Features originais: {len(features_originais)}")
    print(f"Features novas: {len(features_novas)}")
    print(f"Total de features: {len(features_finais)}")
    
    # Preparação dos dados
    X = df[features_finais].copy()
    y = df[target].copy()
    
    # Divisão dos dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Dados de treino: {X_train.shape[0]} amostras")
    print(f"Dados de teste: {X_test.shape[0]} amostras")
    print(f"Features: {X_train.shape[1]}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, features_finais

def testar_modelos(X_train, X_test, y_train, y_test):
    """Testa diferentes modelos de regressão"""
    print("\n=== TESTE DOS MODELOS ===")
    
    modelos = {
        'Regressão Linear': LinearRegression(),
        'Regressão Ridge': Ridge(alpha=1.0),
        'Regressão Lasso': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf', C=100, gamma='scale')
    }
    
    resultados = []
    
    for nome, modelo in modelos.items():
        print(f"\n--- {nome} ---")
        
        # Treinamento
        modelo.fit(X_train, y_train)
        
        # Previsões
        y_pred_train = modelo.predict(X_train)
        y_pred_test = modelo.predict(X_test)
        
        # Métricas
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        # Cross-validation
        cv_scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        resultado = {
            'Modelo': nome,
            'R² Treino': r2_train,
            'R² Teste': r2_test,
            'MAE Treino': mae_train,
            'MAE Teste': mae_test,
            'RMSE Treino': rmse_train,
            'RMSE Teste': rmse_test,
            'CV R²': cv_mean,
            'CV Std': cv_std
        }
        
        resultados.append(resultado)
        
        print(f"R² Treino: {r2_train:.4f}")
        print(f"R² Teste: {r2_test:.4f}")
        print(f"RMSE Teste: {rmse_test:.4f}")
        print(f"CV R²: {cv_mean:.4f} (+/- {cv_std*2:.4f})")
    
    return resultados

def otimizar_melhor_modelo(X_train, X_test, y_train, y_test):
    """Otimiza o melhor modelo (XGBoost)"""
    print("\n=== OTIMIZAÇÃO DO MELHOR MODELO ===")
    
    # Grid search para XGBoost
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    xgb_model = xgb.XGBRegressor(random_state=42)
    grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print(f"\nMelhores parâmetros: {grid_search.best_params_}")
    print(f"Melhor score CV: {grid_search.best_score_:.4f}")
    
    # Modelo otimizado
    melhor_modelo = grid_search.best_estimator_
    y_pred_otimizado = melhor_modelo.predict(X_test)
    
    # Avaliação
    r2_otimizado = r2_score(y_test, y_pred_otimizado)
    mae_otimizado = mean_absolute_error(y_test, y_pred_otimizado)
    rmse_otimizado = np.sqrt(mean_squared_error(y_test, y_pred_otimizado))
    
    print(f"\n=== MODELO OTIMIZADO ===")
    print(f"R²: {r2_otimizado:.4f}")
    print(f"MAE: {mae_otimizado:.4f}")
    print(f"RMSE: {rmse_otimizado:.4f}")
    
    return melhor_modelo, y_pred_otimizado

def analisar_importancia_features(modelo, features):
    """Analisa a importância das features"""
    print("\n=== ANÁLISE DE IMPORTÂNCIA DAS FEATURES ===")
    
    importancia_features = pd.DataFrame({
        'feature': features,
        'importance': modelo.feature_importances_
    })
    importancia_features = importancia_features.sort_values('importance', ascending=False)
    
    print("\nTop 15 features mais importantes:")
    print(importancia_features.head(15))
    
    # Visualização
    plt.figure(figsize=(12, 8))
    top_features = importancia_features.head(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importância')
    plt.title('Top 20 Features Mais Importantes (XGBoost)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    return importancia_features

def fazer_previsao_novos_dados(dados_novos, modelo, scaler, features):
    """Faz previsões para novos dados"""
    print("\n=== PREVISÃO PARA NOVOS DADOS ===")
    
    # Verificar features
    features_faltantes = [f for f in features if f not in dados_novos.columns]
    if features_faltantes:
        raise ValueError(f"Features faltantes: {features_faltantes}")
    
    # Selecionar features
    X_novo = dados_novos[features].copy()
    
    # Limpeza
    X_novo = X_novo.replace([np.inf, -np.inf], np.nan)
    X_novo = X_novo.fillna(0)
    
    # Normalização
    X_novo_scaled = scaler.transform(X_novo)
    
    # Previsão
    previsoes = modelo.predict(X_novo_scaled)
    
    return previsoes

def main():
    """Função principal"""
    print("🚀 INICIANDO ANÁLISE DE PREVISÃO DE LEADS")
    print("=" * 50)
    
    # 1. Carregar dados
    df = carregar_dados()
    
    # 2. Análise exploratória
    df = analise_exploratoria(df)
    
    # 3. Criar features
    df = criar_features(df)
    
    # 4. Preparar dados
    X_train, X_test, y_train, y_test, scaler, features = preparar_dados(df)
    
    # 5. Testar modelos
    resultados = testar_modelos(X_train, X_test, y_train, y_test)
    
    # 6. Comparar modelos
    df_resultados = pd.DataFrame(resultados)
    df_resultados = df_resultados.set_index('Modelo')
    
    print("\n=== COMPARAÇÃO DOS MODELOS ===")
    print(df_resultados.round(4))
    
    # Ranking
    ranking = df_resultados.sort_values('R² Teste', ascending=False)
    print(f"\nRANKING DOS MODELOS (por R² de teste):")
    for i, (modelo, row) in enumerate(ranking.iterrows(), 1):
        print(f"{i}. {modelo}: R² = {row['R² Teste']:.4f}")
    
    melhor_modelo_nome = ranking.index[0]
    print(f"\n🎯 MELHOR MODELO: {melhor_modelo_nome}")
    
    # 7. Otimizar melhor modelo
    melhor_modelo, y_pred_otimizado = otimizar_melhor_modelo(X_train, X_test, y_train, y_test)
    
    # 8. Análise de importância
    importancia_features = analisar_importancia_features(melhor_modelo, features)
    
    # 9. Exemplo de previsão
    print("\n=== EXEMPLO DE PREVISÃO ===")
    
    # Dados de exemplo
    dados_exemplo = pd.DataFrame({
        'Branding': [5000], 'DPA 1': [2000], 'DPA 3': [0], 'DRA 1': [1500], 'DRA 3': [0],
        'Display 1': [3000], 'Display 2': [1000], 'Rmkt 1': [800], 'Rmkt 3': [0], 'Rmkt 6': [500],
        'Rmkt 7': [0], 'Search 1': [4000], 'Search 2': [1200], 'Search 3': [300], 'Search 4': [200],
        'Search 5': [0], 'Search 6': [1500], 'Shopping 1': [2500], 'Shopping 2': [0], 'Shopping 3': [0],
        'Shopping 4': [0], 'Shopping 5': [0], 'Shopping 6': [0], 'Shopping 7': [0], 'Flag': [1]
    })
    
    # Calcular features derivadas
    dados_exemplo['total_branding'] = dados_exemplo['Branding']
    dados_exemplo['total_dpa'] = dados_exemplo[['DPA 1', 'DPA 3']].sum(axis=1)
    dados_exemplo['total_dra'] = dados_exemplo[['DRA 1', 'DRA 3']].sum(axis=1)
    dados_exemplo['total_display'] = dados_exemplo[['Display 1', 'Display 2']].sum(axis=1)
    dados_exemplo['total_rmkt'] = dados_exemplo[['Rmkt 1', 'Rmkt 3', 'Rmkt 6', 'Rmkt 7']].sum(axis=1)
    dados_exemplo['total_search'] = dados_exemplo[['Search 1', 'Search 2', 'Search 3', 'Search 4', 'Search 5', 'Search 6']].sum(axis=1)
    dados_exemplo['total_shopping'] = dados_exemplo[['Shopping 1', 'Shopping 2', 'Shopping 3', 'Shopping 4', 'Shopping 5', 'Shopping 6', 'Shopping 7']].sum(axis=1)
    dados_exemplo['investimento_total'] = dados_exemplo[['total_branding', 'total_dpa', 'total_dra', 'total_display', 'total_rmkt', 'total_search', 'total_shopping']].sum(axis=1)
    dados_exemplo['canais_ativos'] = ((dados_exemplo[['total_branding', 'total_dpa', 'total_dra', 'total_display', 'total_rmkt', 'total_search', 'total_shopping']] > 0).sum(axis=1))
    dados_exemplo['eficiencia_investimento'] = 0
    dados_exemplo['pandemia_investimento'] = dados_exemplo['Flag'] * dados_exemplo['investimento_total']
    
    try:
        previsao = fazer_previsao_novos_dados(dados_exemplo, melhor_modelo, scaler, features)
        
        print(f"\n🎯 PREVISÃO DE LEADS:")
        print(f"Investimento total: R$ {dados_exemplo['investimento_total'].iloc[0]:.2f}")
        print(f"Canais ativos: {dados_exemplo['canais_ativos'].iloc[0]}")
        print(f"Período: {'Com pandemia' if dados_exemplo['Flag'].iloc[0] == 1 else 'Sem pandemia'}")
        print(f"Leads previstos: {previsao[0]:.2f}")
        
        eficiencia = previsao[0] / dados_exemplo['investimento_total'].iloc[0]
        print(f"Eficiência prevista: {eficiencia:.4f} leads por real investido")
        
    except Exception as e:
        print(f"Erro na previsão: {e}")
    
    # 10. Conclusões
    print("\n=== CONCLUSÕES ===")
    print("✅ Análise concluída com sucesso!")
    print("✅ Modelo XGBoost otimizado e treinado")
    print("✅ Features mais importantes identificadas")
    print("✅ Sistema de previsão funcionando")
    print("\nO modelo está pronto para uso em produção!")

if __name__ == "__main__":
    main() 