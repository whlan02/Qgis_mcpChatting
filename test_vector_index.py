from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# Load OpenAI API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize embedding model
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Load FAISS vectorstore
try:
    vectorstore = FAISS.load_local(
        "qgis_vector_index",
        embedding,
        allow_dangerous_deserialization=True
    )
except Exception as e:
    print(f"❌ Failed to load FAISS index: {e}")
    exit()

# Check how many documents are stored
try:
    total_docs = vectorstore.index.ntotal
    print(f"📦 FAISS index loaded successfully with {total_docs} vectors.\n")
except Exception as e:
    print(f"⚠️ Could not check vector count: {e}")

# Test query to check retrieval quality
test_query = "How do I use the raster calculator?"

try:
    print(f"🔍 Running similarity search for:\n   '{test_query}'\n")
    results = vectorstore.similarity_search(test_query, k=3)

    if not results:
        print("❗ No documents retrieved. Check your embeddings or vectorstore content.")
    else:
        print("📚 Top retrieved documents:\n")
        for i, doc in enumerate(results, 1):
            content_preview = doc.page_content[:300].strip().replace("\n", " ")
            print(f"{i}. {content_preview}...\n")
except Exception as e:
    print(f"❌ Error during similarity search: {e}")