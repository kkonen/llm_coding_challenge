# from langchain_scripts.chatbot import chain

# question = "Gebe mir alle Leuchtmittel mit mindestens 1500W und einer Lebensdauer von mehr als 1000 Stunden."

# print(chain.invoke(question))

from langchain_scripts.chatbot import agent_executor


agent_executor.invoke({"input": "Welche Leuchten haben 3000 Watt?"})