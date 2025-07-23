from dotenv import load_dotenv
import os

import pandas as pd
import requests
from langchain_groq import ChatGroq

from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.agents import create_pandas_dataframe_agent

load_dotenv()  # Carrega as variáveis do .env

API_KEY = os.getenv("API_KEY")

llm = ChatGroq(
    temperature=0,
    model="llama-3.3-70b-versatile",
  #  model="llama3.1-70b-8192",
    api_key=API_KEY
)


df = pd.read_csv('/workspaces/langchain/dados_entregas.csv')
print(len(df))


agente_executor = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    agent_type="tool-calling",
    verbose=True,
    allow_dangerous_code=True
)

agente_executor.invoke({"input": """Quais as dimensões do dataframe?
Quais colunas temos e quais os tipos de dados?
"""})

agente_executor.invoke({"input": """Qual o número de linhas do arquivo?
"""})



# Cria uma ferramenta PythonAstREPLTool para executar código Python dinâmico

