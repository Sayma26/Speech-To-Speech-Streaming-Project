import os
import dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.chains import ConversationChain

# 📂 Load API keys from .env
dotenv.load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL")  # e.g., gemini-1.5-flash
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Your Gemini API key

# 🧠 Initialize your smart assistant with Gemini
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.7)

# 💾 Memory module to remember your tasks, names, and more
memory = ConversationEntityMemory(llm=llm, return_messages=True)

# 🔗 Setup the conversation chain with memory
assistant_chain = ConversationChain(
    llm=llm,
    verbose=True,
    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    memory=memory
)

# 🗣️ Begin your productive day!
print("\n📅 Smart Productivity Assistant Activated!")
print("📝 You can talk about tasks, plans, ideas, or just chat.")
print("Type 'reset memory' to clear the chat memory.")
print("Type 'exit' to end the conversation.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("\n👋 Goodbye! Stay productive!")
        break
    elif user_input.lower() == "reset memory":
        memory.clear()
        print("🔄 Memory cleared! Let's start fresh.\n")
        continue
    else:
        reply = assistant_chain.predict(input=user_input)
        print("🤖 Assistant:", reply)
