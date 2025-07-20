# 🤖 Website Chat AI

A Streamlit application that allows you to chat with any website's content using Google Gemini AI and LangChain's RAG (Retrieval-Augmented Generation) technology.

## ✨ Features

- 🌐 **Website Content Loading**: Load and process content from any website URL
- 🧠 **AI-Powered Chat**: Chat with website content using Google Gemini
- 💬 **Chat History**: Maintains conversation history during the session
- 🔍 **Smart Retrieval**: Uses vector embeddings for accurate content retrieval
- 📱 **Responsive Design**: Works great on desktop and mobile

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Environment Variables

1. Copy the environment template:
   ```bash
   cp .env.template .env
   ```

2. Edit `.env` file and add your API keys:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   LANGCHAIN_API_KEY=your_langchain_api_key_here
   ```

### 3. Get Google API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key to your `.env` file

### 4. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🎯 How to Use

1. **Enter Website URL**: Input any website URL in the sidebar
2. **Process Website**: Click "Process Website" to analyze the content
3. **Start Chatting**: Ask questions about the website content
4. **Get AI Responses**: Receive detailed, context-aware answers

## 🛠️ Technical Details

### Architecture

The app uses a RAG (Retrieval-Augmented Generation) pipeline:

1. **Document Loading**: Uses `WebBaseLoader` to fetch website content
2. **Text Splitting**: Breaks content into manageable chunks using `RecursiveCharacterTextSplitter`
3. **Embeddings**: Creates vector embeddings using Google's embedding model
4. **Vector Store**: Stores embeddings in FAISS for fast similarity search
5. **Retrieval Chain**: Combines retrieval and generation for accurate responses

### Key Components

- **LLM**: Google Gemini 1.5 Flash for text generation
- **Embeddings**: Google Generative AI Embeddings
- **Vector Store**: FAISS for efficient similarity search
- **Framework**: LangChain for orchestrating the RAG pipeline
- **UI**: Streamlit for the web interface

## 🎨 Features Overview

### Beautiful UI
- Gradient backgrounds and modern styling
- Responsive design that works on all devices
- Intuitive chat interface with message bubbles
- Status indicators and loading animations

### Smart Processing
- Automatic content chunking for optimal retrieval
- Vector embeddings for semantic search
- Context-aware AI responses
- Error handling and validation

### User Experience
- Real-time chat interface
- Chat history preservation
- Clear status indicators
- Example websites to try

## 🔧 Customization

### Styling
The app uses custom CSS with gradient backgrounds. You can modify the styling in the `st.markdown()` sections of `app.py`.

### AI Parameters
Adjust the LLM parameters in the `initialize_llm()` function:
- `temperature`: Controls response creativity (0-1)
- `max_output_tokens`: Maximum response length

### Chunking Strategy
Modify the text splitting parameters in `load_and_process_website()`:
- `chunk_size`: Size of each text chunk
- `chunk_overlap`: Overlap between chunks

## 📋 Requirements

- Python 3.8+
- Google API Key
- Internet connection for website loading

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

Built with ❤️ using Streamlit, LangChain, and Google Gemini AI
