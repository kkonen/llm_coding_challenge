import glob
import dotenv
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_cohere import CohereEmbeddings
from langchain_cohere import ChatCohere
from langchain.prompts import  PromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
import pandas as pd
from tqdm import tqdm

### Extracts Data from PDF to JSON via LLM

# loading the cohere api key
dotenv.load_dotenv()

PDF_FILES = glob.glob("data/*.pdf")
chat_model = ChatCohere(temperature=0)

template_str = """Deine Aufgabe ist es, aus folgenden Informationen JSON Objekte zu erstelen, 
welche die wichtigsten Informationen über das Produkt enthalten. 
Hier ein Beispiel einer validen JSON, eine JSON ist nur valide, wenn die Varialbennamen und Aufbau identisch sind:
"{{
"produkt": "XBO 1600 W/HSC XL OFR",
"familie": "XBO for cinema projection | Xenon-Kurzbogenlampen 450...10.000 W",
"vorteile": [
    "Kurzbogen mit sehr hoher Leuchtdichte für hellere Leinwandausleuchtung",
    "Konstante Farbtemperatur von 6.000 K über die gesamte Lebensdauer der Lampe",
    "Einfach zu warten",
    "Hohe Lichtbogenstabilität",
    "Sofort Licht auf der Leinwand dank Heißwiederzündungsfunktion",
    "Breiter Dimmbereich"
],
"eigenschaften": {{
    "Farbtemperatur": "6.000 K (Daylight)",
    "Leistung": "1.600 W",
    "Farbwiedergabeindex": "Ra >"
}},
"anwendung": [
    "Klassische 35-mm-Filmprojektion",
    "Digitale Film- und Videoprojektion",
    "Architektur- und Effektlicht (Light Finger)",
    "Sonnensimulation"
],
"daten": {{
    "elektrisch": {{
    "Nennstrom": "65,00 A",
    "Stromsteuerbereich": "50...70 A",
    "Nennspannung": "23,0 V"
    }},
    "abmessungen": {{
    "Durchmesser": "46,0 mm",
    "Länge": "236,0 mm",
    "Länge mit Sockel jedoch ohne Sockelstift": "222,00 mm",
    "Abstand Lichtschwerpunkt (LCL)": "95,0 mm",
    "Kabel-/Leitungslänge, Eingangsseite": "265 mm",
    "Elektrodenabstand kalt": "3,8 mm",
    "Produktgewicht": "321,00 g",
    "Kabellänge": "265,"
    }},
    "temperatur": {{
    "Max. zulässige Umgebungstemp. Quetschung": "230 °C"
    }},
    "lebensdauer": "2500 h",
    "sockel": {{
    "Anode": "SK27/50",
    "Kathode": "SFcX27-8"
    }}
}},
"einsatz": {{
    "Kühlung": "Forciert",
    "Brennstellung": "s20/p20",
    "Anmerkung": "bei vertikaler Brennlage: Anode (+) oben"
}},
"umwelt": {{
    "Deklarationsdatum": "16-12-2022",
    "Primäre Erzeugnisnummer": "4008321299932 | 4062172031721",
    "Stoff der Kandidatenliste": "Lead",
    "CAS Nr. des Stoffes 1": "7439-92-1",
    "SCIP Deklarationsnummer": "49877f08-b6d6-461c-b64a-018655e8a602 | 7a9913f5-169b-4ced-8b20-4a10b8b6fd73"
}},
"verpackung": {{
    "Produkt-Code": "4062172031721",
    "Bezeichnung": "XBO 1600 W/HSC XL OFR",
    "Verpackungseinheit": "Versandschachtel",
    "Abmessungen": "1410 mm x 184 mm x 180 mm",
    "Volumen": "13.58 dm³",
    "Gewicht brutto": "910.00 g"
}}
}}

Übersetze dazu folgenden Text in eine valide JSON:

{context}
"""

# creating the prompts used for LLM interference
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

# creating the chain
chain = (prompt_template | chat_model)

# empty string to fill with jsons
jsons = ""

# querying the LLM in order to convert each PDF into a json
for file in tqdm(PDF_FILES):
    loader = PyPDFLoader(file)
    document = loader.load()
    answer = chain.invoke({"context": document, "question": "Deine Antwort soll nur die fertige JSON enthalten und nichts weiteres."})
    json_str = answer.content.replace("\n","").replace("```","").replace("```","").replace("json", "")
    jsons += json_str

# saving the jsons to a textfile
with open("data/jsons.txt", "w") as file:
    file.write(jsons)