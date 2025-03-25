import os
import dotenv
from langchain.prompts import PromptTemplate 
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
dotenv.load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL")  # e.g., gemini-1.5-flash
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.7)

# Define a prompt template for e-commerce assistant
TEMPLATE = """
You are a virtual E-commerce Shopping Assistant designed to:
1. Help users find products based on their preferences
2. Suggest the best deals, discounts, or offers
3. Recommend popular or trending items in various categories
4. Provide basic product details and comparisons

You are only allowed to answer questions related to online shopping and products.

NOTE: I am an AI assistant and do not have real-time access to product prices or stock. Please verify details on the actual website before making a purchase.

{input}
"""

# Create a prompt and chain
prompt = PromptTemplate.from_template(TEMPLATE)
chain = prompt | llm

# Take user input
user_input = input("What are you looking for today? ")

# Get the AI's response
response = chain.invoke({"input": user_input})

# Display the response
print(response.content)
