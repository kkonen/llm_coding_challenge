import glob
import dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings

PDF_FILES = glob.glob("data/*.pdf")
DB_PATH = "db"

dotenv.load_dotenv()
cohere_embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")

documents = []
for file in PDF_FILES:
    loader = PyPDFLoader(file)
    documents += loader.load_and_split()

vector_db = Chroma.from_documents(
    documents, cohere_embeddings, persist_directory=DB_PATH
)

print("finished")