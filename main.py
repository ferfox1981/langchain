import pandas as pd
import requests
from langchain_groq import ChatGroq
from langchain_experimental.tools import PythonAstREPLTool
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.prompts import ChatPromptTemplate

API_KEY = "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"

llm = ChatGroq(
    temperature=0,
    model="meta-llama/llama-4-scout-17b-16e-instruct",
 #   model="llama3.1-70b-8192",
    api_key=API_KEY
)

# ai_msg = llm.invoke(
#     """
#     Eu tenho um dataframe chamado 'df' com as colunas 'anos_experiencia_agente' e 'tempo_entrega'.
#     Escreva o código Python com a biblioteca Pandas para calcular a correlação entre as duas colunas.
#     Retorne o Markdown para o trecho de código Python e nada mais.
#     """
# )

#print(ai_msg.content)





df = pd.read_csv('/workspaces/langchain/dados_entregas.csv')
#print(df.columns)

ferramenta_python = PythonAstREPLTool(
    locals={
        'df': df,
    },
)



# mostrando o código Python gerado
#res = ferramenta_python.invoke(
#    "df['anos_experiencia_agente'].corr(df['tempo_entrega'])"
#) 
#print(res)

# fazendo na tora a correlação
#if 'anos_experiencia_agente' in df.columns and 'tempo_entrega' in df.columns:
#    # Calcular a correlação entre as duas colunas
#    correlacao = df['anos_experiencia_agente'].corr(df['tempo_entrega'])
#    print(f"Correlação entre anos_experiencia_agente e tempo_entrega: {correlacao:.2f}")
#else:
#    print("Uma ou ambas as colunas não existem no dataframe.")


#print(df.head())


llm_com_ferramenta = llm.bind_tools([ferramenta_python], tool_choice=ferramenta_python.name)

#resposta = llm_com_ferramenta.invoke(
#    """Eu tenho um dataframe 'df' e quero saber a correlação entre as colunas 'anos_experiencia_agente' 
#    e 'tempo_entrega'"""
#)


parser = JsonOutputKeyToolsParser(key_name=ferramenta_python.name, first_tool_only=True)

#cadeia = llm_com_ferramenta | parser

#cadeia.invoke(
#    """Eu tenho um dataframe 'df' e quero saber a correlação entre as colunas 'anos_experiencia_agente' 
#    e 'tempo_entrega'"""
#)

system = f"""Você tem acesso a um dataframe pandas `df`. \
O dataframe contém as seguintes colunas: {df.columns}. \
Dada uma pergunta do usuário, escreva o código Python para respondê-la. \
Retorne SOMENTE o código Python válido e nada mais. \
Não presuma que você tem acesso a nenhuma biblioteca além das bibliotecas Python integradas e pandas."""

prompt = ChatPromptTemplate.from_messages([("system", system),("human", "{question}")])

cadeia = prompt | llm_com_ferramenta | parser | ferramenta_python

resposta = cadeia.invoke({"question": "Qual é a correlação entre anos de experiência do agente e tempo de entrega?"})
print(resposta)

resposta = cadeia.invoke({"question": "Qual é a média do tempo de entrega para cada tipo de clima?"})
print(resposta)