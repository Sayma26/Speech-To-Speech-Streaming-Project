import os
import dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain

# Load environment variables
dotenv.load_dotenv()

# Get API key and model name
GEMINI_MODEL = os.getenv("GEMINI_MODEL")  # e.g., "gemini-1.5-flash"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # your actual API key

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set in .env file")

# Set the Google API key as an environment variable (important for LangChain)
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.7)

# Set up memory to summarize conversations
memory = ConversationSummaryMemory(llm=llm, return_messages=True)

# Create conversation chain with memory
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

print("\n🤖 AI Chatbot Ready! (type 'exit' to quit, 'reset memory' to clear chat)\n")

# Interactive chat loop
while True:
    user_input = input("You: ")

    if user_input.strip().lower() == "exit":
        print("👋 Goodbye! Have a great day!")
        break

    elif user_input.strip().lower() == "reset memory":
        memory.clear()
        print("🧠 Memory cleared! Start a new conversation.")
        continue

    else:
        try:
            response = conversation.predict(input=user_input)
            print("Gemini:", response)
        except Exception as e:
            print("⚠️ Error:", str(e))
