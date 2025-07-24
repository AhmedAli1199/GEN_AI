from fastapi import FastAPI
from langserve import add_routes
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Initialize the model with proper model name
model = ChatGroq(
    model="llama3-70b-8192",  
    groq_api_key=groq_api_key,
    temperature=0.1
)


prompt = ChatPromptTemplate.from_messages([
    ("system", "Translate any text given to you to French:"),
    ("user", "{input}"),
])

# Create output parser
parser = StrOutputParser()

# Create the chain
chain = prompt | model | parser

# Create FastAPI app
app = FastAPI(
    title="Langserve App", 
    description="A FastAPI app using Langserve and Groq"
)

# Add routes for the chain
add_routes(
    app, 
    chain, 
    path="/chain"
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=4000)
