#from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.output_parsers import JsonOutputKeyToolsParser
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain_community.agent_toolkits.json.toolkit import JsonToolkit

load_dotenv()  # Carrega as variáveis do .env

API_KEY = os.getenv("API_KEY")

llm = ChatGroq(
    temperature=0,
    model="meta-llama/llama-4-scout-17b-16e-instruct",
 #   model="llama3.1-70b-8192",
    api_key=API_KEY
)

# Ferramenta que retorna um JSON com cidade e temperatura
@tool
def pegar_clima(cidade: str) -> dict:
    """Retorna o clima de uma cidade em JSON"""
    return {
        "cidade": cidade,
        "temperatura": "28°C"
    }

# Lista de ferramentas
tools = [pegar_clima]

# Define um parser que extrai a chave "temperatura" do JSON
parser = JsonOutputKeyToolsParser(key_name="temperatura")

# Cria um toolkit com o parser
toolkit = JsonToolkit(tools=tools, output_parser=parser, spec=)

# Inicializa um agente com as ferramentas e o parser
#llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
agent = initialize_agent(
    tools=toolkit.get_tools(),
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Rodar a query
resposta = agent.run("Qual é a temperatura agora em Salvador?")
print("Temperatura extraída:", resposta)
