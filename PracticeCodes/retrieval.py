import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# Required ENV variables
GEMINI_MODEL = os.getenv("GEMINI_MODEL")  # e.g., "gemini-1.5-flash"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables.")

# Set Google API key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.7)

# Load PDF
print("📄 Loading PDF...")
loader = PyMuPDFLoader("../documents/Artificial_Intelligence.pdf")
documents = loader.load()

# Split into chunks
print("✂️ Splitting text...")
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# Embed chunks
print("🔗 Embedding text...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store in Chroma DB
print("💾 Storing in ChromaDB...")
chroma_db = Chroma.from_documents(docs, embeddings)

# Create Retriever
retriever = chroma_db.as_retriever()

# Build QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# Define Tools
tools = [
    Tool(
        name="Chroma Retriever",
        func=qa_chain.run,
        description="Useful for answering questions about the uploaded document."
    )
]

# Memory for conversation
memory = ConversationBufferMemory(memory_key="chat_history")

# Initialize agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
    memory=memory
)

# Chat Loop
print("\n🤖 Ask questions about the document (type 'exit' to quit):")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ['exit', 'quit']:
        print("👋 Bye!")
        break
    response = agent.run(user_input)
    print("Gemini:", response)
