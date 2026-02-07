# 🤖 RAG-Based PDF Question Answering

An intelligent PDF Q&A application using **Retrieval-Augmented Generation (RAG)**. Upload documents, get AI-powered answers instantly.

🔗 **[Live Demo](#)** *(Add link after deployment)*

## 🎯 What It Does

Upload any PDF → Ask questions → Get accurate, context-aware answers powered by AI.

**Try it:** Use the built-in demo PDFs or upload your own documents.

## ✨ Key Features

- 📄 Multi-PDF support with real-time preview
- 🔍 Vector similarity search (FAISS)
- 🤖 AI-powered responses (Groq Llama 3.1)
- ⚡ Adjustable parameters (chunk size, temperature, top-k)
- 🎯 Demo mode with sample documents
- 🔒 Rate limiting & security controls

## 🛠️ Tech Stack

**Backend:** Python, FAISS, Sentence Transformers  
**LLM:** Groq API (Llama 3.1 8B Instant)  
**Frontend:** Streamlit  
**NLP:** PyPDF, all-MiniLM-L6-v2 embeddings

## 🚀 Quick Setup
```bash
# Clone repo
git clone https://github.com/yourusername/rag-pdf-qa.git
cd rag-pdf-qa

# Install dependencies
pip install -r requirements.txt

# Add your Groq API key to .env
GROQ_API_KEY=your_key_here

# Run app
streamlit run app.py
```

## 🏗️ How It Works
```
PDF Upload → Text Extraction → Chunking → Embeddings → FAISS Index
                                                            ↓
User Question → Similarity Search → Context Retrieval → LLM Answer
```

## 📸 Screenshots

*(Add 1-2 screenshots here after deployment)*

## 👨‍💻 Author

**Your Name**  
[GitHub](https://github.com/Tusharvimal) • [LinkedIn](https://www.linkedin.com/in/tusharvimal/)

---

⭐ Star this repo if you find it useful!