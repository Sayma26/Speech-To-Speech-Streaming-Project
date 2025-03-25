import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import getpass
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub

# Load environment variables (for Gemini & Pinecone config)
load_dotenv()

# 🌐 Travel Blog Embedding Model (Hugging Face)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 🧭 Set up Pinecone (vector DB for travel knowledge)
if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = getpass.getpass("🔐 Enter your Pinecone API key: ")

pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

# 🌍 Travel Guide Index (where all destination knowledge is stored)
index_name = os.getenv("INDEX_NAME")  # e.g., "travel-guide-index"

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if index_name not in existing_indexes:
    print("Creating a new destination index...")
index = pc.Index(index_name)

# 🗺️ Load the destination data index into LangChain's wrapper
vectorstore = PineconeVectorStore(index, embedding=embeddings)

# 🧠 Load the LLM to generate smart, chatty travel responses
llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL"), temperature=0.7)

# 🧳 Retrieve destination insights from Pinecone
def get_destination_retriever():
    return vectorstore.as_retriever()

# 🧩 Build the AI travel chain
def create_travel_qa_chain(retriever, llm):
    prompt_template = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_chain = create_stuff_documents_chain(llm, prompt_template)
    qa_chain = create_retrieval_chain(retriever, combine_chain)
    return qa_chain

# 💬 Ask travel questions & get Gemini-generated answers
def explore_destination(query, llm):
    retriever = get_destination_retriever()
    qa_chain = create_travel_qa_chain(retriever, llm)
    response = qa_chain.invoke({"input": query})
    return response

# 🌟 User Interaction
user_query = input("📌 Where would you like to explore today? Ask a travel question: ")
response = explore_destination(user_query, llm)
print("\n🗺️ Here's what we found:\n")
print(response)
