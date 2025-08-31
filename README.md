# RAG Chatbot 🧠💬

A document-based chatbot using Retrieval-Augmented Generation (RAG) built with Python, LangChain, FAISS, and OpenAI. Ask questions about your uploaded documents and get precise answers with source citations.

## ✨ Features

- **Document Processing**: Upload and index .txt, .pdf, .docx, .md files
- **Smart Retrieval**: Uses FAISS for fast vector similarity search
- **Embedding Cache**: Optimizes performance by caching embeddings
- **Web Search Fallback**: Optional web search when documents don't contain answers
- **Source Citations**: Shows which documents were used to generate answers
- **Modular Architecture**: Easy to extend or integrate into APIs

## 📁 Project Structure

```
rag_chatbot/
├── src/
│   ├── components/
│   │   ├── chatbot/          # Query handling and response generation
│   │   ├── config/           # Settings and logging configuration
│   │   ├── ingestion/        # Document loading and processing
│   │   ├── retrieval/        # Vector storage, embeddings, web search
│   │   └── prompts/          # LLM prompt templates
│   ├── utils/                # Helper functions and exceptions
│   └── main.py              # Application entry point
├── data/
│   ├── raw_docs/            # Place your documents here
│   ├── db/                  # Vector index and metadata storage
│   └── prompts/             # YAML prompt files
└── logs/                    # Application logs (if file logging enabled)
```

## ⚙️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/rag_chatbot.git
cd rag_chatbot
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Setup environment variables**
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

4. **Add your documents**
```bash
# Place your documents in data/raw_docs/
# Supported formats: .pdf, .docx, .txt, .md
```

## 🚀 Usage

**Start the chatbot:**
```bash
python src/main.py
```

The application will:
- Process documents from `data/raw_docs/`
- Create vector embeddings (cached for performance)
- Start an interactive chat session

**Example interaction:**
```
🔍 Ask me: What are the main topics covered in the documents?
📝 Response: Based on the documents, the main topics include...
📚 Sources (documents):
  1. ../data/raw_docs/document1.pdf
  2. ../data/raw_docs/document2.txt
```

## ⚙️ Configuration

Key settings in your `.env` file:

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for embeddings and LLM | ✅ |
| `SEARCH_API_KEY` | Google Custom Search API key | ❌ |
| `SEARCH_ENGINE_ID` | Google Custom Search Engine ID | ❌ |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | ❌ |
| `IS_FILE_LOGGING_ENABLED` | Enable file-based logging | ❌ |

## 🔧 Advanced Features

- **Embedding Cache**: Automatically caches embeddings to avoid recomputation
- **Web Search**: Falls back to web search when documents don't contain answers
- **Configurable Chunking**: Adjust chunk size and overlap for optimal retrieval
- **Source Attribution**: Always shows which documents were used for answers

## 📋 Commands

- Type `quit`, `exit`, or `bye` to exit the chatbot
- Ctrl+C for immediate shutdown

## 🛠️ Development

The codebase uses a modular architecture:

- **Document Processing**: Handles loading and chunking of various file formats
- **Vector Storage**: FAISS-based similarity search with metadata
- **Embedding Cache**: Redis-like caching for embeddings
- **Query Processing**: LLM-based response generation with retrieval

## 📝 Logging

Logs are written to:
- Console (colored output for development)
- Files in `logs/` directory (when `IS_FILE_LOGGING_ENABLED=True`)

## 🚨 Troubleshooting

**No documents processed?**
- Check that documents are in `data/raw_docs/`
- Verify file formats are supported (.pdf, .docx, .txt, .md)

**API errors?**
- Ensure `OPENAI_API_KEY` is set in `.env`
- Check API quota and billing

**Web search not working?**
- Set `SEARCH_API_KEY` and `SEARCH_ENGINE_ID` for Google Custom Search
- Or leave unset to use fallback search methods
