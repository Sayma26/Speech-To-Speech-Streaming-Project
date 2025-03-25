import os
import dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
dotenv.load_dotenv()
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Ensure the API key is present
if not GOOGLE_API_KEY:
    raise ValueError("Google API key missing! Please check your environment settings.")

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.7)

# Use HuggingFace for document embeddings
embeddings = HuggingFaceEmbeddings()

# 📜 Class for managing the Historical Manuscript Explorer
class HistoricalManuscriptExplorer:
    def __init__(self, archive_directory="./manuscript_archive"):
        self.archive_directory = archive_directory
        self.archive = None

    def load_manuscript(self, manuscript_path):
        """ Load and preprocess a historical manuscript """
        if not os.path.exists(manuscript_path):
            raise FileExistsError(f"❌ Manuscript not found: {manuscript_path}")

        loader = TextLoader(manuscript_path)
        manuscript_docs = loader.load()

        splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separator="\n"
        )

        processed_chunks = splitter.split_documents(manuscript_docs)
        return processed_chunks

    def build_archive(self, documents):
        """ Build a vector archive from historical documents """
        if not documents:
            raise ValueError("⚠️ No manuscript content provided for archiving.")

        self.archive = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=self.archive_directory
        )
        self.archive.persist()
        return self.archive

    def open_archive(self):
        """ Load an existing manuscript archive """
        if not os.path.exists(self.archive_directory):
            raise FileNotFoundError(f"⚠️ Archive not found at {self.archive_directory}")

        self.archive = Chroma(
            persist_directory=self.archive_directory,
            embedding_function=embeddings
        )
        return self.archive

    def explore(self, question, k=3):
        """ Ask questions about the manuscripts """
        if not self.archive:
            raise ValueError("⚠️ Archive not initialized. Please build or open the archive first.")

        retriever = self.archive.as_retriever(search_kwargs={"k": k})

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        result = qa_chain.invoke({"query": question})
        return result

# 🛠️ Usage Example
explorer = HistoricalManuscriptExplorer()
chunks = explorer.load_manuscript("medieval_scrolls.txt")
explorer.build_archive(chunks)
explorer.open_archive()
answer = explorer.explore("What do the manuscripts reveal about medieval trade routes?")
print(answer)
