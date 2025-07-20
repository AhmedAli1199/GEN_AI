import streamlit as st
import os
import asyncio
import nest_asyncio
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
import time
import validators

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

load_dotenv(override=True)

# Page configuration
st.set_page_config(
    page_title="💬 Website Chat AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .chat-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    .user-message {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    
    .bot-message {
        background: linear-gradient(90deg, #a8edea 0%, #fed6e3 100%);
        color: #333;
        padding: 1rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 0;
        max-width: 80%;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    
    .sidebar-content {
        background: linear-gradient(180deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .status-success {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .status-processing {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'retrieval_chain' not in st.session_state:
    st.session_state.retrieval_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_url' not in st.session_state:
    st.session_state.current_url = ""

def setup_environment():
    """Setup environment variables for API keys"""
    try:
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
        os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "default")
        return True
    except Exception as e:
        st.error(f"Error setting up environment: {str(e)}")
        return False

def safe_load_website(url):
    """Safely load website content with proper error handling"""
    try:
        # Method 1: Try standard WebBaseLoader
        loader = WebBaseLoader(url)
        loader.requests_kwargs = {
            'verify': False,
            'timeout': 30,
            'headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        }
        documents = loader.load()
        print("Loaded using standard WebBaseLoader")
        return documents
    except Exception as e1:
        try:
            # Method 2: Try with web_path parameter
            loader = WebBaseLoader(web_path=url)
            documents = loader.load()
            print("Loaded using WebBaseLoader with web_path")
            return documents
        except Exception as e2:
            try:
                # Method 3: Try with different configuration
                loader = WebBaseLoader([url])
                documents = loader.load()
                print("Loaded using WebBaseLoader with list")
                return documents
            except Exception as e3:
                raise Exception(f"Failed to load website. Tried multiple methods. Last error: {str(e3)}")

def initialize_llm():
    """Initialize the LLM model"""
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,
            max_output_tokens=1024
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

def load_and_process_website(url):
    """Load and process website content"""
    try:
        with st.spinner("🌐 Loading website content..."):
            # Use safe loading method
            documents = safe_load_website(url)
            
            if not documents:
                raise Exception("No content could be extracted from the website")
            
        with st.spinner("✂️ Splitting content into chunks..."):
            # Split documents into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            docs = splitter.split_documents(documents)
            
            if not docs:
                raise Exception("No text chunks could be created from the website content")
            
        with st.spinner("🧠 Creating embeddings..."):
            # Create embeddings - using the newer embedding model
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            
            # Create vector store
            vectorstore = FAISS.from_documents(docs, embeddings)
            
        return vectorstore, len(docs)
    except Exception as e:
        st.error(f"Error processing website: {str(e)}")
        st.error("Please check if:")
        st.error("- The URL is accessible and valid")
        st.error("- Your internet connection is stable")
        st.error("- Your Google API key is correctly set")
        return None, 0

def create_chat_chain(vectorstore, llm):
    """Create the retrieval chain for chat"""
    try:
        # Create prompt template
        prompt = ChatPromptTemplate.from_template(
            """Answer the question based on the context below. Be helpful, accurate, and provide detailed responses.

            <context>
            {context}
            </context>

            Question: {input}
            
            Answer:"""
        )
        
        # Create document chain
        document_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=prompt,
        )
        
        # Create retrieval chain
        retriever = vectorstore.as_retriever()
        retrieval_chain = create_retrieval_chain(
            retriever, document_chain
        )
        
        return retrieval_chain
    except Exception as e:
        st.error(f"Error creating chat chain: {str(e)}")
        return None

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Website Chat AI</h1>
        <p>Chat with any website using advanced AI - Powered by Google Gemini & LangChain</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Setup environment
    if not setup_environment():
        st.stop()
    
    # Initialize LLM
    llm = initialize_llm()
    if not llm:
        st.stop()
    
    # Sidebar for URL input and controls
    with st.sidebar:
        st.markdown("### 🌐 Website Input")
        
        # URL input
        url = st.text_input(
            "Enter Website URL:",
            placeholder="https://example.com",
            help="Enter a valid website URL to chat with its content"
        )
        
        # Process button
        if st.button("🚀 Process Website", type="primary"):
            if url and validators.url(url):
                if url != st.session_state.current_url:
                    # Clear previous chat history when new URL is processed
                    st.session_state.chat_history = []
                    st.session_state.current_url = url
                    
                    # Process the website
                    vectorstore, num_chunks = load_and_process_website(url)
                    
                    if vectorstore:
                        st.session_state.vectorstore = vectorstore
                        
                        # Create retrieval chain
                        retrieval_chain = create_chat_chain(vectorstore, llm)
                        if retrieval_chain:
                            st.session_state.retrieval_chain = retrieval_chain
                            
                            st.markdown(f"""
                            <div class="status-success">
                                ✅ Website processed successfully!<br>
                                📄 {num_chunks} content chunks created
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error("Failed to create chat chain")
                    else:
                        st.error("Failed to process website")
                else:
                    st.info("This website is already processed. You can start chatting!")
            else:
                st.error("Please enter a valid URL")
        
        # Status display
        if st.session_state.current_url:
            st.markdown("### 📊 Current Session")
            st.markdown(f"**Website:** {st.session_state.current_url}")
            st.markdown(f"**Status:** {'Ready to chat' if st.session_state.retrieval_chain else 'Not processed'}")
            
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
        
        # Instructions
        st.markdown("""
        ### 📝 How to Use:
        1. **Enter a website URL** in the input field above
        2. **Click "Process Website"** to analyze the content
        3. **Start chatting** with the website content using the chat interface
        4. **Ask specific questions** about the website's content
        """)
    
    # Main chat interface
    if st.session_state.retrieval_chain:
        st.markdown("### 💬 Chat Interface")
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["type"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <strong>You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bot-message">
                    <strong>AI:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
        
        # Chat input
        with st.form("chat_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                user_question = st.text_input(
                    "Ask a question about the website:",
                    placeholder="What is this website about?",
                    label_visibility="collapsed"
                )
            with col2:
                submit_button = st.form_submit_button("Send 📤", type="primary")
        
        if submit_button and user_question:
            # Add user message to chat history
            st.session_state.chat_history.append({
                "type": "user",
                "content": user_question
            })
            
            # Get AI response
            with st.spinner("🤔 AI is thinking..."):
                try:
                    response = st.session_state.retrieval_chain.invoke({
                        "input": user_question
                    })
                    
                    ai_response = response['answer']
                    
                    # Add AI response to chat history
                    st.session_state.chat_history.append({
                        "type": "ai",
                        "content": ai_response
                    })
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error getting AI response: {str(e)}")
    
    else:
        # Welcome message
        st.markdown("""
        <div class="chat-container">
            <h3>👋 Welcome to Website Chat AI!</h3>
            <p>To get started:</p>
            <ul>
                <li>Enter a website URL in the sidebar</li>
                <li>Click "Process Website" to analyze the content</li>
                <li>Start chatting with the website content!</li>
            </ul>
            <p><em>This AI-powered tool uses Google Gemini and LangChain to help you chat with any website's content.</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Example URLs
        st.markdown("### 🌟 Try these example websites:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📚 Google AI Docs"):
                st.session_state.example_url = "https://ai.google.dev/gemini-api/docs/embeddings"
        
        with col2:
            if st.button("🤖 OpenAI Blog"):
                st.session_state.example_url = "https://openai.com/blog"
        
        with col3:
            if st.button("📖 Wikipedia"):
                st.session_state.example_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"

if __name__ == "__main__":
    main()
