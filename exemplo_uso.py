#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exemplo de uso do modelo de previsão de leads
Este arquivo demonstra como usar o modelo treinado para fazer previsões
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

def carregar_modelo():
    """Carrega o modelo treinado (simulação)"""
    print("Carregando modelo...")
    # Em um cenário real, você carregaria o modelo salvo
    # modelo = pickle.load(open('modelo_leads.pkl', 'rb'))
    # scaler = pickle.load(open('scaler_leads.pkl', 'rb'))
    
    print("Modelo carregado com sucesso!")
    return None, None

def criar_dados_exemplo():
    """Cria dados de exemplo para teste"""
    print("\n=== DADOS DE EXEMPLO ===")
    
    # Dados de investimento em marketing
    dados = {
        'Branding': [8000],
        'DPA 1': [3000],
        'DPA 3': [0],
        'DRA 1': [2500],
        'DRA 3': [0],
        'Display 1': [5000],
        'Display 2': [2000],
        'Rmkt 1': [1200],
        'Rmkt 3': [0],
        'Rmkt 6': [800],
        'Rmkt 7': [0],
        'Search 1': [6000],
        'Search 2': [1800],
        'Search 3': [500],
        'Search 4': [300],
        'Search 5': [0],
        'Search 6': [2200],
        'Shopping 1': [4000],
        'Shopping 2': [0],
        'Shopping 3': [0],
        'Shopping 4': [0],
        'Shopping 5': [0],
        'Shopping 6': [0],
        'Shopping 7': [0],
        'Flag': [1]  # 1 = com pandemia, 0 = sem pandemia
    }
    
    df = pd.DataFrame(dados)
    
    # Calcular features derivadas
    df['total_branding'] = df['Branding']
    df['total_dpa'] = df[['DPA 1', 'DPA 3']].sum(axis=1)
    df['total_dra'] = df[['DRA 1', 'DRA 3']].sum(axis=1)
    df['total_display'] = df[['Display 1', 'Display 2']].sum(axis=1)
    df['total_rmkt'] = df[['Rmkt 1', 'Rmkt 3', 'Rmkt 6', 'Rmkt 7']].sum(axis=1)
    df['total_search'] = df[['Search 1', 'Search 2', 'Search 3', 'Search 4', 'Search 5', 'Search 6']].sum(axis=1)
    df['total_shopping'] = df[['Shopping 1', 'Shopping 2', 'Shopping 3', 'Shopping 4', 'Shopping 5', 'Shopping 6', 'Shopping 7']].sum(axis=1)
    df['investimento_total'] = df[['total_branding', 'total_dpa', 'total_dra', 'total_display', 'total_rmkt', 'total_search', 'total_shopping']].sum(axis=1)
    df['canais_ativos'] = ((df[['total_branding', 'total_dpa', 'total_dra', 'total_display', 'total_rmkt', 'total_search', 'total_shopping']] > 0).sum(axis=1))
    df['eficiencia_investimento'] = 0  # Será calculada após a previsão
    df['pandemia_investimento'] = df['Flag'] * df['investimento_total']
    
    print("Dados de exemplo criados:")
    print(f"Investimento total: R$ {df['investimento_total'].iloc[0]:.2f}")
    print(f"Canais ativos: {df['canais_ativos'].iloc[0]}")
    print(f"Período: {'Com pandemia' if df['Flag'].iloc[0] == 1 else 'Sem pandemia'}")
    
    return df

def fazer_previsao(dados, modelo, scaler, features):
    """Faz a previsão de leads"""
    print("\n=== FAZENDO PREVISÃO ===")
    
    try:
        # Selecionar features
        X = dados[features].copy()
        
        # Limpeza
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # Normalização
        X_scaled = scaler.transform(X)
        
        # Previsão
        previsao = modelo.predict(X_scaled)
        
        return previsao[0]
        
    except Exception as e:
        print(f"Erro na previsão: {e}")
        return None

def analisar_resultado(previsao, dados):
    """Analisa o resultado da previsão"""
    if previsao is not None:
        print(f"\n🎯 RESULTADO DA PREVISÃO:")
        print(f"Leads previstos: {previsao:.2f}")
        
        investimento_total = dados['investimento_total'].iloc[0]
        eficiencia = previsao / investimento_total
        
        print(f"\n📊 ANÁLISE:")
        print(f"Investimento total: R$ {investimento_total:.2f}")
        print(f"Eficiência: {eficiencia:.4f} leads por real investido")
        print(f"Custo por lead: R$ {investimento_total/previsao:.2f}")
        
        # Recomendações
        print(f"\n💡 RECOMENDAÇÕES:")
        if eficiencia > 0.01:
            print("✅ Eficiência acima da média - continue com a estratégia atual")
        else:
            print("⚠️ Eficiência abaixo da média - considere otimizar os canais")
        
        if dados['canais_ativos'].iloc[0] >= 5:
            print("✅ Boa diversificação de canais")
        else:
            print("⚠️ Considere diversificar mais os canais de marketing")
            
        if dados['Flag'].iloc[0] == 1:
            print("📅 Período de pandemia - ajuste as expectativas de conversão")
        else:
            print("📅 Período normal - performance esperada padrão")

def main():
    """Função principal"""
    print("🚀 EXEMPLO DE USO DO MODELO DE PREVISÃO DE LEADS")
    print("=" * 60)
    
    # 1. Carregar modelo (simulação)
    modelo, scaler = carregar_modelo()
    
    # 2. Criar dados de exemplo
    dados = criar_dados_exemplo()
    
    # 3. Fazer previsão
    # Em um cenário real, você usaria o modelo carregado
    # previsao = fazer_previsao(dados, modelo, scaler, features)
    
    # Simulação da previsão
    print("\n⚠️ SIMULAÇÃO - Modelo não carregado")
    print("Para usar o modelo real, execute primeiro o arquivo 'analise_leads_completa.py'")
    
    # Simular uma previsão baseada nos dados
    investimento_total = dados['investimento_total'].iloc[0]
    canais_ativos = dados['canais_ativos'].iloc[0]
    flag_pandemia = dados['Flag'].iloc[0]
    
    # Fórmula simplificada para simulação
    previsao_simulada = (investimento_total * 0.001) * (1 + canais_ativos * 0.1) * (0.8 if flag_pandemia else 1.0)
    
    print(f"\n🎯 PREVISÃO SIMULADA:")
    print(f"Leads previstos: {previsao_simulada:.2f}")
    
    # 4. Analisar resultado
    analisar_resultado(previsao_simulada, dados)
    
    print("\n✅ Exemplo concluído!")
    print("Para usar o modelo real, execute 'analise_leads_completa.py' primeiro")

if __name__ == "__main__":
    main() 