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

Change the question you want to ask inside the ```main.py``` file and run it via  
```python main.py```