import dotenv
from langchain_cohere import ChatCohere
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import  PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
dotenv.load_dotenv()

DB_PATH = "db/"

chat_model = ChatCohere(temperature=0)

vector_db = Chroma(persist_directory=DB_PATH,
    embedding_function=CohereEmbeddings(model="embed-multilingual-v3.0"))

retriever  = vector_db.as_retriever(k=10)

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

chain = ({"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | chat_model
    | StrOutputParser())