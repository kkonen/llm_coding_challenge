# from langchain_scripts.chatbot import chain

# question = "Gebe mir alle Leuchtmittel mit mindestens 1500W und einer Lebensdauer von mehr als 1000 Stunden."
# question = "Welche Leuchten haben 3000 Watt?"
# print(chain.invoke(question))

from langchain_scripts.chatbot import agent_executor

agent_executor.invoke({"input": "Gebe mir alle Leuchtmittel mit mindestens 1000W und einer Lebensdauer von mehr als 2000 Stunden."})