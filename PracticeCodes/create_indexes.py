import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Name of the vector index for storing book memory
vault_name = os.getenv("VAULT_NAME")  # e.g., "book-recommender-index"

# Check if the memory vault already exists
available_vaults = [vault_info["name"] for vault_info in pinecone_client.list_indexes()]

if vault_name not in available_vaults:
    # Create the memory vault (index)
    pinecone_client.create_index(
        name=vault_name,
        dimension=384,  # Matching HF embedding output size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        deletion_protection="enabled"
    )
    
    # Wait until the vault is ready
    while not pinecone_client.describe_index(vault_name).status["ready"]:
        time.sleep(1)
    
    print("📚 Memory Vault created successfully.")
else:
    print("✅ Vault already exists. Skipping creation.")

# Connect to the memory vault
vault_index = pinecone_client.Index(vault_name)

# Initialize vector store for storing book embeddings
vector_store = PineconeVectorStore(
    index=vault_index,
    embedding=HuggingFaceEmbeddings()
)

print("📖 Book Memory Vault is ready for recommendations!")
