from dotenv import load_dotenv
import os

import pandas as pd
import requests
from langchain import hub
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import Tool
from langchain_experimental.tools import PythonAstREPLTool
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
from IPython.display import Markdown, display
import matplotlib.pyplot as plt
import seaborn as sns




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

# Essa é a ferramenta exploradora
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

############################
#relatorio_informacoes = informacoes_dataframe.invoke("quais as informaçoes gerais sobre o dataframe?")
#display(Markdown(relatorio_informacoes))
############################

@tool
def resumo_estatístico(pergunta: str) -> str:
    """Utilize esta ferramenta sempre que o usuário solicitar um resumo estatístico do DataFrame,
    incluindo média, mediana, desvio padrão, valores mínimo e máximo, quartis, contagem de dados nulos, etc..."""
    estatisticas_descritivas = df.describe(include='number').transpose().to_string()
    template_resposta = PromptTemplate(
        template="""Você é um analista de dados encarregado de interpretar resultados estatísticos de uma base de dados
a partir de uma {pergunta} feita pelo usuário.

A seguir, você encontrará as estatísticas descritivas da base de dados:
============ESTATÍSTICAS DESCRITIVAS============
{resumo}
================================================

Com base nesses dados, elabore um resumo explicativo com linguagem clara, acessível e fluída, destacando
os principais pontos dos resultados. Inclua:
1. Um título: ## Relatório de estatísticas descritivas;
2. Uma visão geral das estatísticas das colunas numéricas;
3. Um parágrafo sobre cada uma das colunas, comentando informações sobre seus valores;
4. Identificação de possíveis outliers com base nos valores mínimo e máximo;
5. Recomendações de próximos passos na análise com base nos padrões identificados.
""",
        input_variables=['pergunta', 'resumo']
    )
    cadeia = template_resposta | llm | StrOutputParser()
    resposta = cadeia.invoke({"pergunta": pergunta, "resumo": estatisticas_descritivas})
    return resposta

############################
#relatorio_estatisticas = resumo_estatístico.invoke("quais as estatísticas descritivas dos dados?")
#display(relatorio_estatisticas)
############################

@tool
def gerar_grafico(pergunta: str) -> str:  
    """Utilize esta ferramenta sempre que a pessoa usuária solicitar um gráfico a partir de um 
      DataFrame PandasDef, com base em uma instrução do usuário. A instrução pode conter pedidos como:
      crie um gráfico de média de tempo de entrega por clima, plote a distribuição do tempo de entrega, 
      ou plote a relação entre a classificação dos agentes e o tempo de entrega. Palavras-chave comuns que indicam 
      o uso dessa ferramenta incluem: crie um gráfico, plote, visualize, faça um gráfico de, mostre a distribuição, 
      represente graficamente, exiba, entre outros."""
    colunas_info = "\n".join([f"- {col} ({dtype})" for col, dtype in df.dtypes.items()])
    amostra_dados = df.head(3).to_dict(orient='records')
    template_resposta = PromptTemplate(
    template = """
        Você é um especialista em visualização de dados. Sua tarefa é gerar *apenas o código Python**
        para plotar um gráfico com base na solicitação do usuário.

        ## Solicitação do usuário:
        "{pergunta}"

        ## Metadados do DataFrame:
        {colunas}

        ## Amostra dos dados (3 primeiras linhas):
        {amostra}

        ## Instruções obrigatórias:
        1. Use as bibliotecas `matplotlib.pyplot` (como `plt`) e `seaborn` (como `sns`);
        2. Defina o tema com `sns.set_theme()`;
        3. Certifique-se de que todas as colunas mencionadas na solicitação existem no DataFrame chamado `df`;
        4. Escolha o tipo de gráfico adequado conforme a análise solicitada:
            - **Distribuição de variáveis numéricas**: `histplot`, `kdeplot`, `boxplot` ou `violinplot`
            - **Distribuição de variáveis categóricas**: `countplot`
            - **Comparação entre categorias**: `barplot`
            - **Relação entre variáveis**: `scatterplot` ou `lineplot`
            - **Séries temporais**: `lineplot`, com o eixo X formatado como datas
        5. Configure o tamanho do gráfico com `figsize=(8,4)`;
        6. Adicione título e rótulos (`labels`) apropriados aos eixos, deixando um espaço extra para não cortar os rótulos;
        7. Posicione o título à esquerda com `loc='left'`, deixe o `pad=20` e use `fontsize=14`;
        8. Mantenha os ticks eixo X sem rotação com `plt.xticks(rotation=0)`;
        9. Remova as bordas superior e direita do gráfico com `sns.despine()`;
        10. Finalize o código com `plt.show()`.

        Retorne APENAS o código Python, sem nenhum texto adicional ou explicação.

        Código Python:```
        """,
            input_variables=["pergunta", "colunas", "amostra"]
        )
    cadeia = template_resposta | llm | StrOutputParser()
    codigo_bruto = cadeia.invoke({
    "pergunta": pergunta,
    "colunas": colunas_info,
    "amostra": amostra_dados
    })
    codigo_limpo = codigo_bruto.replace("```python", "").replace("```", "").strip()
    

    exec_globals = {"df": df, "plt": plt, "sns": sns}
    exec_locals = {}
    exec(codigo_limpo, exec_globals, exec_locals)
    fig = plt.gcf()
    return fig


############################
#fig = gerar_grafico.invoke("crie um gráfico de média de tempo de entrega por clima");   
#fig.savefig("/workspaces/langchain/figura_tempo_entrega_por_clima.png")
#resposta_grafico_1 = gerar_grafico.invoke("""Crie um gráfico da média do tempo de entrega por clima.
#Ordene do maior para o menor valor utilizando uma paleta de cores""")
#resposta_grafico_1.savefig("/workspaces/langchain/figura_tempo_entrega_por_clima1.png")
#resposta_grafico_2 = gerar_grafico.invoke("""Plote um boxplot do tempo de entrega.""")
#resposta_grafico_2.savefig("/workspaces/langchain/figura_tempo_entrega_por_clima2.png")
##############################

ferramenta_informacoes_dataframe = Tool(
    name="Informações DataFrame",
    func=informacoes_dataframe,
    description="""Utilize esta ferramenta sempre que o usuário solicitar informações gerais sobre o dataframe,
                incluindo número de colunas e linhas, nomes das colunas e seus tipos de dados, contagem de dados nulos e
                duplicados para dar um panorama geral sobre o arquivo.""",return_direct=True
)

ferramenta_resumo_estatistico = Tool(
    name="Resumo Estatístico",
    func=resumo_estatístico,
    description="""Utilize esta ferramenta sempre que o usuário solicitar um resumo estatístico completo
            e descritivo da base de dados, incluindo várias estatísticas (média, desvio padrão, mínimo, máximo etc.).
            Não utilize esta ferramenta para calcular uma única métrica como 'qual é a média de X' ou 
            qual a correlação das variáveis'. Nesses casos, utilize a ferramenta_codigos_python.""",
    return_direct=True
)

ferramenta_gerar_grafico = Tool(
    name="Gerar Gráfico",
    func=gerar_grafico,
    description="""Utilize esta ferramenta sempre que o usuário solicitar um gráfico a partir de um DataFrame
            pandas ('df') com base em uma instrução do usuário. A instrução pode conter pedidos como: 'Crie um gráfico
            da média de tempo de entrega por clima', 'Plote a distribuição do tempo de entrega' ou 'Plote a relação
            entre a classificação dos agentes e o tempo de entrega'. Palavras-chave comuns que indicam o uso desta
            ferramenta incluem: 'crie um gráfico', 'plote', 'visualize', 'faça um gráfico de', 'mostre a distribuição',
            'represente graficamente', entre outros.""",return_direct=True
)

ferramenta_codigos_python = Tool(
    name="Códigos Python",
    func=PythonAstREPLTool(locals={"df": df}),
    description="""Utilize esta ferramenta sempre que o usuário solicitar cálculos, consultas ou transformações
        específicas usando Python diretamente sobre o DataFrame `df`.
        Exemplos de uso incluem: "Qual é a média da coluna X?", "Quais são os valores únicos da coluna Y?",
        "Qual a correlação entre A e B?". Evite utilizar esta ferramenta para solicitações mais amplas ou descritivas,
        como informações gerais sobre o DataFrame, resumos estatísticos completos ou geração de gráficos - nesses casos,
        use as ferramentas apropriadas.""", return_direct=True
)

prompt_react = hub.pull("hwchase17/react")
#print(prompt_react.template)

df_head = df.head().to_markdown()

prompt_react_pt = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
    partial_variables={"df_head": df_head},
    template = """Você é um assistente que sempre responde em português.

Você tem acesso a um dataframe pandas chamado `df`.
Aqui estão as primeiras linhas do DataFrame, obtidas com `df.head().to_markdown()`:

{df_head}

Responda às seguintes perguntas da melhor forma possível.

Para isso, você tem acesso às seguintes ferramentas:

{tools}

Use o seguinte formato:

Question: a pergunta de entrada que você deve responder
Thought: você deve sempre pensar no que fazer
Action: a ação a ser tomada, deve ser uma das [{tool_names}]
Action Input: a entrada para a ação
Observation: o resultado da ação
... (este Thought/Action/Action Input/Observation pode se repetir N vezes)
Thought: Agora eu sei a resposta final
Final Answer: a resposta final para a pergunta de entrada original.

Comece!

Question: {input}
Thought: {agent_scratchpad}"""
)

tools = [
    ferramenta_informacoes_dataframe,
    ferramenta_resumo_estatistico,
    ferramenta_gerar_grafico,
    ferramenta_codigos_python
]

agente = create_react_agent(llm=llm, tools=tools, prompt=prompt_react_pt)

orquestrador = AgentExecutor(agent=agente, tools=tools, verbose=True, handle_parsing_errors=True)

resposta = orquestrador.invoke({"input": "Gostaria de um relatório com informações sobre o dataframe"})
display(resposta["output"])

resposta2 = orquestrador.invoke({"input": "quero saber as estatísticas descritivas dos dados"})
display(resposta2["output"])

#resposta3 = orquestrador.invoke({"input": "Crie um gráfico da média do tempo de entrega por clima. Ordene do maior para o menor valor."})
#fig = resposta3["output"]
#if hasattr(fig, "savefig"):
#    fig.savefig("/workspaces/langchain/fig_agente_pensante.png")

resposta4 = orquestrador.invoke({"input": "Qual é a média do tempo_entrega ?"})    
display(resposta4["output"])


