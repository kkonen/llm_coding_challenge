import dotenv
from langchain_cohere import ChatCohere
from langchain.agents import AgentExecutor
from langchain.agents import Tool
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import  PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_scripts.query_json import query_jsons

# loading cohere api key
dotenv.load_dotenv()

# specifing vectordatabase path
DB_PATH = "db/"

# loading the chatmodel
chat_model = ChatCohere(temperature=0)

# loading the vectordatabase
vector_db = Chroma(persist_directory=DB_PATH,
    embedding_function=CohereEmbeddings(model="embed-multilingual-v3.0"))

retriever  = vector_db.as_retriever(k=10)

# creating the prompt templates 
template_str = """Deine Aufgabe ist es Fragen zu Produktdatenblättern von OSRAM Kurzbogenlampen zu beantworten.
Erfinde keine zusätzlichen Informationen. Wenn etwas unklar ist, frage nach spezifischeren Fragen.

{context}
"""

system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["context"],
        template=template_str))

human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"],
        template="{question}"))

prompt_template = ChatPromptTemplate(
    input_variables=["context", "question"],
    messages=[system_prompt, human_prompt])

# creating the retriever chain
chain = ({"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | chat_model
    | StrOutputParser())

# creates the retriever tool to be used by the agent
retriever_tool = Tool(name="retriever_tool", 
                      description="Beantwortet einfache Fragen wie z.B. Welche Leuchte hat 3000W? oder Was ist die Lebensdauer von Leuchte X?",
                      func=chain.invoke)
# creates the json query tool to be used by the agent
json_tool = Tool(name="json_tool", 
                      description="Beantwortet komplexere Anfragen wie z.B. Gebe mir alle Leuchtmittel mit mindestens 1500W und einer Lebensdauer von mehr als 3000 Stunden.",
                      func=query_jsons)

# creating the agent prompt and agent executor
agent_prompt = ChatPromptTemplate.from_template("{input}")

agent = create_cohere_react_agent(llm=chat_model,
   tools=[retriever_tool, json_tool],
   prompt=agent_prompt)

agent_executor = AgentExecutor(agent=agent,
    tools=[retriever_tool, json_tool], verbose=True)