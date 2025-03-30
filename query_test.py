from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
import os

# Load .env for your OpenAI key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize embedding model
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Load FAISS vectorstore (you already created this in previous step)
vectorstore = FAISS.load_local(
    "qgis_vector_index",
    embedding,
    allow_dangerous_deserialization=True
)

# Set up a Q&A chain using OpenAI + your vector DB
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=openai_api_key),
    retriever=vectorstore.as_retriever()
)

# Ask a sample question (you can loop this later)
query = "How can I load this vector layer in QGIS and select only the polygon where the name is Bratislava?"
response = qa_chain.invoke(query)

# Print clean answer
print("\n🧠 QGIS Bot Answer:")
print(f"\n❓ Question: {response['query']}")
print(f"\n💡 Answer:\n{response['result']}")
