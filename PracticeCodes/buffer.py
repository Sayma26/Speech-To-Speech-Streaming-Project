import os
import dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain

# Load environment variables from .env file
dotenv.load_dotenv()

# Load Gemini model name and API key from .env
GEMINI_MODEL = os.getenv("GEMINI_MODEL")  # e.g., gemini-1.5-flash
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Set Google API key for LangChain
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.7)

# Set up short-term memory (last 2 interactions)
memory = ConversationBufferWindowMemory(k=2, return_messages=True)

# Create a conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Chat loop
print("\n💬 Welcome to Gemini Chatbot! (type 'exit' to quit, 'reset memory' to clear memory)\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("👋 Goodbye! Have a great day!")
        break
    elif user_input.lower() == "reset memory":
        memory.clear()
        print("🧹 History cleaned! Start a new conversation.")
        continue
    else:
        response = conversation.predict(input=user_input)
        print(f"🤖 Gemini: {response}\n")
