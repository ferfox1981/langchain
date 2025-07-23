from dotenv import load_dotenv
import os

import pandas as pd
import requests
from langchain_groq import ChatGroq
from langchain_experimental.tools import PythonAstREPLTool
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, ToolMessage
from typing import Optional
from langchain_core.tools import tool


load_dotenv()  # Carrega as variáveis do .env

API_KEY = os.getenv("API_KEY")

chat = ChatGroq(
    temperature=0,
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=API_KEY
)


prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente amigável."), 
    ("human", "{input}"),
    ])


#cadeia = prompt | chat 

#response = cadeia.invoke({"input": "Qual a temperatura em Porto Alegre?"})
#print(response.content)




@tool
def temperatura_atual(location: str, unit: Optional[str]):
    """Retorna a temperatura de uma determinada localidade
    
    Args:
        location (str): Nome da localidade
        unit (Optional[str]): Unidade de medida da temperatura, por exemplo, "C" para Celsius ou "F" para Fahrenheit.
    Returns:
        str: Temperatura atual na localidade especificada.
    """
    return "25ºC"

@tool
def lugar_externo(location: str, unit: Optional[str]):
    """Retorna se há lugar externo na localidade
    Args:
        location (str): Nome da localidade
        unit (Optional[str]): Unidade de medida da temperatura, por exemplo, "C" para Celsius ou "F" para Fahrenheit.
    Returns:
        str: Mensagem indicando se há um lugar externo disponível.
    """
    return "Há um local externo disponível"

tools = [temperatura_atual, lugar_externo]

llm_com_ferramentas = chat.bind_tools(tools)

#parser = JsonOutputKeyToolsParser(key_name="temperatura_atual", first_tool_only=True)

#cadeia = prompt | tool_model | parser 

messages = [HumanMessage(content="Eu gostaria de saber se a temperatura de Porto Alegre é 25ºC ou mais. Se sim, eu gostaria de saber se há um lugar externo.")]

llm_output = llm_com_ferramentas.invoke(messages)

print(llm_output)

messages.append(llm_output)

tool_mapping = {
    "temperatura_atual": temperatura_atual,
    "lugar_externo": lugar_externo
}

for tool_call in llm_output.tool_calls:
    tool = tool_mapping[tool_call["name"].lower()]
    tool_output = tool.invoke(tool_call["args"])
    messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

print(messages)