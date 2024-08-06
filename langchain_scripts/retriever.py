import glob
import dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings

# path to vectordatabase and pdf files
PDF_FILES = glob.glob("data/*.pdf")
DB_PATH = "db"

#loading cohere api key
dotenv.load_dotenv()
# creating cohere embedder
cohere_embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")

# iterating over all pdfs and converting them to documents
documents = []
for file in PDF_FILES:
    loader = PyPDFLoader(file)
    documents += loader.load_and_split()

# creating the vectordb with embeddings of the documents
vector_db = Chroma.from_documents(
    documents, cohere_embeddings, persist_directory=DB_PATH
)

print("finished")