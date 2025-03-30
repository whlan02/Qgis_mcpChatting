from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Path to your converted text files
docs_path = "/Users/moritzdenk/Geoinformatics/Uni/ifgi_hack/mcpchatting/training/converted_txt"

# Load .txt files recursively
loader = DirectoryLoader(docs_path, glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()

# Split into 500-token chunks with 50-token overlap
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# Create embeddings

# Store in FAISS vector DB
vectorstore = FAISS.from_documents(chunks, embedding)

# Save locally
vectorstore.save_local("qgis_vector_index")