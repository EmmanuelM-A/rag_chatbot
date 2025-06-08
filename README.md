# RAG Chatbot 🧠💬

A document-based chatbot using Retrieval-Augmented Generation (RAG) built with Python, LangChain, FAISS, and OpenAI 
(or local LLMs). Ask questions about your uploaded documents and get precise answers!

## ✨ Features

- Upload and index .txt, .pdf, .docx, .md files
- Ask natural questions and get grounded answers
- Uses FAISS for local vector search
- Modular structure (easy to extend or plug into an API)
- Optional usage tracking + logging

## 📁 Folder Structure

```commandline
    rag_chatbot/
    ├── components/         # Core logic: embedding, document loading, etc.
    ├── data/               # Vector index, uploaded docs
        ├── db/             # Stores vectors and metadata
        ├── raw_docs/       # Stores documents
    ├── prompts/            # System + user prompt templates
    ├── utils/              # Logging
```

## ⚙️ Installation

1. **Clone the repo.**

```commandline
git clone https://github.com/yourusername/rag_chatbot.git
```
2. **Change directory into the project.**

```commandline
cd rag_chatbot
```
3. **Install libraries from requirements.**

```commandline
pip install -r requirements.txt
```

4. **Setup `.env` file. In the `.env` you must set your OpenAI API Key.**

```commandline
cp .env.example .env
```


## 🚀 Quick Start

```commandline
python components/main.py
```

## Future Roadmap

See [Future Enhancements](FUTURE.md) for planned features like:
- API Extension
- Usage limits per user
- Frontend extension
- Support for local LLMs (via Ollama, HuggingFace)
