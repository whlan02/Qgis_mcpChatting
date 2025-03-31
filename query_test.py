from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

# Load env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Embeddings + Vectorstore
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = FAISS.load_local("qgis_vector_index", embedding, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# LLMs
llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=openai_api_key)
general_llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=openai_api_key)

### Prompt for question rephrasing
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

### Prompt for QA
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

### Chat History Store
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

### Full Chain with memory
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

### Helper function with fallback
def smart_invoke(query, session_id):
    try:
        result = conversational_rag_chain.invoke(
            {"input": query},
            config={"configurable": {"session_id": session_id}}
        )
        answer = result.get("answer", "")

        if not answer or "I don't know" in answer.lower():
            print("🟡 Source: GPT fallback (no relevant RAG context found)")
            fallback = general_llm.invoke(query)
            return fallback.content
        else:
            print("🟢 Source: RAG with memory")
            return answer

    except Exception as e:
        print(f"🔴 Error during RAG: {e}")
        print("🟡 Source: GPT fallback (error in RAG)")
        return general_llm.invoke(query).content

### Start chat loop
print("\n💬 RAG Chat with Memory + GPT Fallback\n")
session_id = "demo-session"

while True:
    query = input("🧑 You: ")
    if query.lower() in ["exit", "quit", "q"]:
        print("👋 Exiting chat. Goodbye!")
        break
    response = smart_invoke(query, session_id)
    print(f"\n🤖 AI: {response}\n")