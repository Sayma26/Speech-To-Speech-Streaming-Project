import os
import dotenv
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory

# Load environment variables
dotenv.load_dotenv()

# Retrieve Gemini model and Google API key from .env
GEMINI_MODEL = os.getenv("GEMINI_MODEL")  # Example: gemini-1.5-flash
GOOGLE_API_KEY = os.getenv("AIzaSyA5aJWBBZ12TVUgdN3D-v7orr4vQY_ouEo")

# Initialize Google Gemini LLM
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.7)

# Define a simple calculator tool
def calculator_tool(query: str):
    try:
        return eval(query)
    except:
        return "Invalid math expression."

calc_tool = Tool(
    name="Calculator",
    func=calculator_tool,
    description="Solves basic math expressions."
)

# Initialize memory for chat history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize the LangChain agent
agent = initialize_agent(
    tools=[calc_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# Chat loop
print("🤖 Chatbot is running! Type 'exit' to stop.\n")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Chatbot: Goodbye! 👋")
        break
    response = agent.run(user_input)
    print("Chatbot:", response)
