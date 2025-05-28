import pandas as pd
import requests
from langchain_groq import ChatGroq

API_KEY = "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"

llm = ChatGroq(
    temperature=0,
    model="meta-llama/llama-4-scout-17b-16e-instruct",
 #   model="llama3.1-70b-8192",
    api_key=API_KEY
)

ai_msg = llm.invoke(
    """
    Eu tenho um dataframe chamado 'df' com as colunas 'anos_experiencia_agente' e 'tempo_entrega'.
    Escreva o código Python com a biblioteca Pandas para calcular a correlação entre as duas colunas.
    Retorne o Markdown para o trecho de código Python e nada mais.
    """
)

#print(ai_msg.content)





df = pd.read_csv('/workspaces/langchain/dados_entregas.csv')

if 'anos_experiencia_agente' in df.columns and 'tempo_entrega' in df.columns:
    # Calcular a correlação entre as duas colunas
    correlacao = df['anos_experiencia_agente'].corr(df['tempo_entrega'])
    print(f"Correlação entre anos_experiencia_agente e tempo_entrega: {correlacao:.2f}")
else:
    print("Uma ou ambas as colunas não existem no dataframe.")

#print(df.head())