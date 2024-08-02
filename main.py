from langchain_scripts.chatbot import chain

question = "Welche Lampen haben einen Stromsteuerbereich von 70 bis 110 Ampere?"

print(chain.invoke(question))

