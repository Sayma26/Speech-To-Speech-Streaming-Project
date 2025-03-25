import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

if __name__ == "__main__":

    # 1️⃣ Loading Documents
    print("📂 Loading Documents...")
    doc_path = os.path.join(os.getcwd(), "documents", "documents.txt")  # Ensure correct path
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"❌ Error: The file {doc_path} does not exist!")

    loader = TextLoader(doc_path)
    document = loader.load()  # loads the document with metadata
    print(f"✅ Loaded {len(document)} document(s)")

    # 2️⃣ Splitting Documents
    print("✂️ Splitting Documents...")
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = splitter.split_documents(document)
    print(f"✅ Split into {len(split_documents)} chunks")

    # 3️⃣ Embedding Documents
    print("🔄 Generating Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4️⃣ Inserting Documents into VectorDB
    print("🗂️ Inserting Documents into Pinecone VectorDB...")
    index_name = os.getenv("INDEX_NAME")
    if not index_name:
        raise ValueError("❌ Error: INDEX_NAME is not set in environment variables.")

    vector_db = PineconeVectorStore.from_documents(split_documents, embeddings, index_name=index_name)
    print(f"✅ Successfully inserted {len(split_documents)} chunks into Pinecone VectorDB")
