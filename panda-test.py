import pandas as pd

df = pd.read_csv('/workspaces/langchain/dados_entregas.csv')
print(df.head())  # Exibe as primeiras linhas do dataframe


if 'anos_experiencia_agente' in df.columns and 'tempo_entrega' in df.columns:
    # Calcular a correlação entre as duas colunas
    correlacao = df['anos_experiencia_agente'].corr(df['tempo_entrega'])
    print(f"Correlação entre anos_experiencia_agente e tempo_entrega: {correlacao}")
else:
    print("Uma ou ambas as colunas não existem no dataframe.")

