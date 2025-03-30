from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load vectorstore
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = FAISS.load_local(
    "qgis_vector_index",
    embedding,
    allow_dangerous_deserialization=True
)

# Create the RAG QA chain
llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=openai_api_key)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Simple loop
print("Ask me anything about QGIS (type 'exit' to quit):\n")
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break
    answer = qa_chain.invoke(query)
    print(f"\n🔍 QGIS Bot: {answer}\n")
