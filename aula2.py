from dotenv import load_dotenv
import os

import pandas as pd
import requests
from langchain_groq import ChatGroq
from langchain_experimental.tools import PythonAstREPLTool
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.agents import create_pandas_dataframe_agent

load_dotenv()  # Carrega as variáveis do .env

API_KEY = os.getenv("API_KEY")

llm = ChatGroq(
    temperature=0,
    model="meta-llama/llama-4-scout-17b-16e-instruct",
 #   model="llama3.1-70b-8192",
    api_key=API_KEY
)







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
Não presuma que você tem acesso a nenhuma biblioteca além das bibliotecas Python integradas e pandas. \
Responda em português à pergunta quando tiver informaçõpes suficientes para respondê-la"""

prompt = ChatPromptTemplate.from_messages([("system", system),("human", "{question}"),
                                            MessagesPlaceholder("chat_history", optional=True)])

def _get_history(x:dict) -> list:
    """Analise a saída da cadeia até este ponto em uma lista de mensagens do histórico de chat para inserir \
        no prompt."""
    ai_msg = x["ai_msg"]
    tool_call_id = x["ai_msg"].additional_kwargs["tool_calls"][0]["id"]
    tool_msg = ToolMessage(tool_call_id=tool_call_id, content=str(x["tool_output"]))
    return [ai_msg, tool_msg]


cadeia = (
         
          RunnablePassthrough().assign(ai_msg=prompt | llm_com_ferramenta)
         .assign(tool_output=itemgetter("ai_msg") | parser | ferramenta_python)
         .assign(chat_history=_get_history)
         .assign(response=prompt | llm | StrOutputParser())
         .pick(["tool_output", "response"])
         )

#resposta = cadeia.invoke({"question": "Qual é a correlação entre anos de experiência do agente e tempo de entrega?"})
#print(resposta)

resposta_1 = cadeia.invoke({"question": "Qual é a média do tempo de entrega para cada tipo de clima?"})
print(resposta_1['response'])

resposta_2 = cadeia.invoke({"question": "Qual é a média do tempo de entrega?"})
print(resposta_2['response'])

resposta_3 = cadeia.invoke({"question": "Qual é a mediana da mesma coluna?"})
print(resposta_3['response'])

resposta_4 = cadeia.invoke({"question": """Qual é a correlação entre anos de experiencia do agente
e tempo de entrega?
É maior que a correlação entre classificacao do agente e tempo de entrega?
"""})
print(resposta_4["response"])

