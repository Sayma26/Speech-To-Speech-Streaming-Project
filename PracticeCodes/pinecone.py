import os
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, DirectoryLoader, UnstructuredFileLoader, Docx2txtLoader
)
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Function to choose appropriate loader based on file extension
def load_all_documents(directory_path):
    loaders = [
        DirectoryLoader(directory_path, glob="*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(directory_path, glob="*.docx", loader_cls=Docx2txtLoader),
        DirectoryLoader(directory_path, glob="*.txt", loader_cls=TextLoader),
        DirectoryLoader(directory_path, glob="*", loader_cls=UnstructuredFileLoader),  # For other file types
    ]

    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    return documents

if __name__ == "__main__":
    # 1️⃣ Loading Documents
    print("📂 Loading directory documents...")
    directory_path = "../documents"
    
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"❌ Error: Directory '{directory_path}' does not exist.")
    
    documents = load_all_documents(directory_path)
    print(f"✅ Documents Loaded: {len(documents)}")

    # 2️⃣ Splitting Documents
    print("\n✂️ Splitting Documents...")
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = splitter.split_documents(documents)
    print(f"✅ Split {len(documents)} documents into {len(split_documents)} chunks.")

    # 3️⃣ Embedding Documents
    print("\n🔄 Generating Embeddings...")
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("✅ Embedding Completed.")

    # 4️⃣ Inserting into Pinecone VectorDB
    print("\n📦 Inserting Documents into Pinecone VectorDB...")
    index_name = os.getenv("INDEX_NAME")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    if not index_name or not pinecone_api_key:
        raise EnvironmentError("❌ Error: INDEX_NAME or PINECONE_API_KEY is missing in .env file.")

    vector_db = PineconeVectorStore.from_documents(
        split_documents,
        embedding,
        index_name=index_name,
        pinecone_api_key=pinecone_api_key
    )

    print(f"✅ Successfully inserted {len(split_documents)} chunks into Pinecone VectorDB.")
