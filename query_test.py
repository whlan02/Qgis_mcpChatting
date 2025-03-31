from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from dotenv import load_dotenv
import os

# Load OpenAI key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set up embedding and vectorstore
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = FAISS.load_local(
    "qgis_vector_index",
    embedding,
    allow_dangerous_deserialization=True
)

# Create base chain (without memory)
base_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=openai_api_key),
    retriever=vectorstore.as_retriever(),
    return_source_documents=False
)

# Add memory via RunnableWithMessageHistory
chain_with_history = RunnableWithMessageHistory(
    base_chain,
    lambda session_id: InMemoryChatMessageHistory(),  # You can replace with persistent memory if needed
    input_messages_key="question",
    history_messages_key="chat_history"
)

# Start chat
print("\n💬 RAG Chat with New Memory Style. Type 'exit' to quit.\n")

# You can assign a session ID (e.g., for saving multiple users)
session_id = "user-session-1"

while True:
    query = input("🧑 You: ")
    if query.lower() in ["exit", "quit", "q"]:
        print("👋 Exiting chat. Goodbye!")
        break

    response = chain_with_history.invoke(
    {"question": query},
    config={"configurable": {"session_id": session_id}}
)

    print(f"\n🤖 AI: {response['answer']}\n")

    print("📚 Retrieved source documents:")
    for doc in response.get("source_documents", []):
        print(f"- {doc.page_content[:300]}...\n")
