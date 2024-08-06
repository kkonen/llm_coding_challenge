import glob
import dotenv
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_cohere import CohereEmbeddings
from langchain_cohere import ChatCohere
from langchain.prompts import  PromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
import pandas as pd

# loading cohere api key
dotenv.load_dotenv()

def query_jsons(user_question):
    """
    Parameters:
    user_question (string): The question the user wants to ask

    Returns:
    string: The LLMs answer to the question

    Example:
    >>> query_jsons("Gebe mir alle Leuchtmittel mit mindestens 1500W und einer Lebensdauer von mehr als 1000 Stunden.")
    "Folgende Leuchtmittel entsprechen Ihren Kriterien:

    - XBO 4000 W/HTP XL OFR

    - XBO 4000 W/HTP XL OFR
    - XBO 4000 W/HTP XL OFR
    - XBO 4000 W/HS XL OFR
    - XBO 4000 W/HS XL OFR
    - XBO 4500 W/HS XL OFR
    - XBO 5000 W/H XL OFR
    - XBO 6000 W/HS XL OFR
    - XBO 7000 W/HS XL OFR
    - XBO 10000 W/HS OFR"
    """

    # Load the textfile containing the jsons with information about the products
    with open("data/jsons.txt", 'r') as file:
        jsons = file.read()
    
    chat_model = ChatCohere(temperature=0)
    
    # Create the prompt for LLM query
    template_str = """Deine Aufgabe ist es, die folgenden JSONs zu benutzen, um Fragen über die Produkte zu beantworten.

    {context}"""
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

    # Creates the Chain for LLM query
    chain = (prompt_template | chat_model)

    # Invoking the chain to get an answer using the jsons
    answer = chain.invoke({"context": jsons, "question": user_question})
    
    return answer