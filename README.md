# Installation

Create a new Python virtual environment with  
```python -m venv venv```

Activate the environment:  
Linux ```source venv/bin/activate```  
Windows ```venv\Scripts\activate```

Install necessary dependencies:  
```pip install -r requirements.txt```

Add a ```.env``` file to the root folder and add your Cohere API Key.
We are using free LLMs from Cohere (https://cohere.com/)

Lastly, place your PDFs inside the ```data/``` folder

# Usage

If you are starting from scratch, first run:  
```python langchain_scripts/retriever.py```  
in order to create the vector database needed for our semantic search of the supplied PDFs.

Also run the json extractor, in order to ask questions like the ones specified for task 2 (this may take some time):
```python langchain_scripts/create_jsons.py```

Change the question you want to ask inside the ```main.py``` file and run it via  
```python main.py```

# Fragen Teil Zwei:

Funktioniert deine Lösung auch für 500 PDFs?
* Nein, da bei 500 PDFs vermutlich der RAM überläuft.  

Was gibt es bei so vielen PDFs zu beachten? 
* Man müsste das Embedden und Abspeichern der VectorDB in kleineren Batches vornehmen, eventuell sollte man darüber nachdenken PineCone, Faiss oder eine andere Vectordatenbank, die speziell für größere Mengen an Daten ausgelegt ist, anstatt ChromaDB zu nutzen. Ebenso wäre eine paralelisierte Datenverarbeitung hilfreich.
Dazu kommt, dass bei einer größeren Anzahl von Embeddings die Zeit für das Durchsuchen und Finden der relevantesten Embeddings zunehmen kann.

Wo sind bottlenecks oder etwaige Probleme zu erwarten?
* Etwaige Bottlenecks können wie oben genannt der RAM Speicher sein, oder das Erstellen der Embeddings.
Die Geschwindigkeit, mit der Embeddings in die Datenbank geschrieben werden, kann ein Bottleneck sein. Optimierungen wie Batch-Insertions können hier helfen.
Zusätzlich könnte der Festplattenspeicher, bei sehr sehr vielen PDFs irgendwann ein Bottleneck darstellen, bei einer Anzahl von 500 PDFs sollte dies jedoch noch kein Problem darstellen. Wie oben genannt, kann ebenfalls die benötigte Zeit zum Durchsuchen der Embeddings zunehmen und ggf. ein Bottleneck darstellen.
Ein Zusätzlicher Bottleneck ist sicherlich der Json Extractor, hier ist bereits bei wenigen PDFs mit einer großen Wartezeit zu rechnen.
# Docker 

1. Place the PDFs inside the ```data/``` folder.
2. Create a ```.env``` file inside the root folder and add your ```COHERE_API_KEY``` 
3. Run ```docker build -t llm_challenge .```
4. Attach to the docker with ```docker run -it llm_challenge bash```
5. See section Usage 