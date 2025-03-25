import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Set your document directory
DOCUMENT_PATH = r"../documents"  # Adjust if needed

# Load all PDF files in the directory
loader = DirectoryLoader(DOCUMENT_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

# Split each document into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = []

for doc in docs:
    split_texts = text_splitter.split_text(doc.page_content)
    for text in split_texts:
        split_docs.append(Document(page_content=text, metadata=doc.metadata))

print(f"✅ Total split documents: {len(split_docs)}")

# Initialize HuggingFace embeddings
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Setup ChromaDB and persist path
DB_PATH = "chroma_db"
db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)

# Add documents in batches
batch_size = 500
total_batches = (len(split_docs) + batch_size - 1) // batch_size

for i in range(0, len(split_docs), batch_size):
    batch = split_docs[i:i + batch_size]
    db.add_documents(batch)
    print(f"📦 Inserted batch {i // batch_size + 1} / {total_batches}")

print("✅ Embedded document store created successfully!")
