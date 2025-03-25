import os
import dotenv
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Load environment variables
dotenv.load_dotenv()

# Get model and API key
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.7)

# Set up memory for ongoing conversation
memory = ConversationBufferMemory(memory_key="travel_chat_history", return_messages=True)

# Custom prompt for a travel assistant
TEMPLATE = """
You are a friendly AI Travel Assistant who helps users plan vacations, suggest destinations, activities, and tips based on their interests and preferences.

Conversation History:
{travel_chat_history}

User: {input}
"""

prompt = PromptTemplate(input_variables=["travel_chat_history", "input"], template=TEMPLATE)

# Set up the conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=True,
)

# Interactive loop
print("\n🌍 Welcome to your AI Travel Assistant!")
print("Type 'exit' to quit or 'reset memory' to start fresh.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("✈️ Safe travels! Goodbye!")
        break
    elif user_input.lower() == "reset memory":
        memory.clear()
        print("🧳 Memory cleared. Ready for a new travel adventure!")
        continue
    else:
        response = conversation.predict(input=user_input)
        print("TravelBot:", response)
