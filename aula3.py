from dotenv import load_dotenv
import os

import pandas as pd
import requests
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from IPython.display import Markdown, display


load_dotenv()  # Carrega as variáveis do .env

API_KEY = os.getenv("API_KEY")

llm = ChatGroq(
    temperature=0,
    model="meta-llama/llama-4-scout-17b-16e-instruct",
 #   model="llama3.1-70b-8192",
    api_key=API_KEY
)


df = pd.read_csv('/workspaces/langchain/dados_entregas.csv')
#print(df.apply(lambda col: col[~col.isna()].astype(str).str.strip().str.lower().eq('nan')).sum())


@tool
def informacoes_dataframe(pergunta: str) -> str:
    """Utilize esta ferramenta sempre que o usuário solicitar informações gerais sobre o DataFrame,
    incluindo número de colunas e linhas, nomes das colunas e seus tipos de dados, contagem de dados nulos
    e duplicados para dar um panorama geral sobre o arquivo."""
    shape = df.shape
    columns = df.dtypes
    nulos = df.isnull().sum()
    nans_str = df.apply(lambda col: col[~col.isna()].astype(str).str.strip().str.lower().eq('nan')).sum()
    duplicados = df.duplicated().sum()

    template_resposta = PromptTemplate(
    template = """
Você é um analista de dados encarregado de apresentar um resumo informativo sobre um DataFrame a partir de uma {pergunta} feita pelo usuário.

A seguir, você encontrará as informações gerais da base de dados:
==================== INFORMAÇÕES DO DATAFRAME ====================
Dimensões: {shape} da forma mais detalhada possível, ou seja, número de linhas e colunas.

Colunas e tipos de dados:
{columns}

Valores nulos por coluna:
{nulos}

Strings 'nan' (qualquer capitalização) por coluna:
{nans_str}

Linhas duplicadas: {duplicados}
==================================================================

Com base nessas informações, escreva um resumo claro e organizado contendo:
1. Um título: ## Relatório de Informações gerais sobre o dataset,
2. A dimensão total do DataFrame,
3. A descrição de cada coluna (incluindo nome, tipo de dado e o que aquela coluna é),
4. As colunas que contém dados nulos, com a respectiva quantidade;
5. As colunas que contém strings 'nan', com a respectiva quantidade;
6. E a existência (ou não) de dados duplicados;
7. Escreva um parágrafo sobre analises que podem ser feitas com esses dados;
8. Escreva um parágrafo sobre tratamentos que podem ser feitos nos dados.
""",
    input_variables=['pergunta', 'shape', 'columns', 'nulos', 'nans_str', 'duplicados']
)
    cadeia = template_resposta | llm | StrOutputParser()

    resposta = cadeia.invoke({
    'pergunta': pergunta,
    'shape': shape,
    'columns': columns,
    'nulos': nulos,
    'nans_str': nans_str,
    'duplicados': duplicados
})
    return resposta

relatorio_informacoes = informacoes_dataframe.invoke("quais as informaçoes gerais sobre o dataframe?")
display(Markdown(relatorio_informacoes))