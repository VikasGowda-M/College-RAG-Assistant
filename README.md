# 🎓 College RAG Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers college-related questions using PDF documents.

This project uses **Sentence Transformers** for embeddings, **ChromaDB** as the vector database, and **Google Gemini** as the Large Language Model (LLM).

---

## 🚀 Features

- 📄 Reads multiple college PDF documents
- ✂️ Splits documents into chunks
- 🧠 Generates embeddings using Sentence Transformers
- 🗄️ Stores embeddings in ChromaDB
- 🔍 Retrieves the most relevant document chunks
- 🤖 Uses Google Gemini to generate accurate answers
- 📚 Shows the source document and page number
- 💬 Supports continuous chatbot conversation

---

## 🛠️ Tech Stack

- Python
- Google Gemini API
- Sentence Transformers
- ChromaDB
- LangChain Text Splitters
- PyPDF
- Python Dotenv

---

## 📂 Project Structure

```
College-RAG-Assistant/
│
├── data/
│   ├── Attendance.pdf
│   ├── Fees.pdf
│   ├── Library.pdf
│   └── ...
│
├── app.py
├── requirements.txt
├── README.md
├── .env
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/VikasGowda-M/College-RAG-Assistant.git
```

Go to the project folder

```bash
cd College-RAG-Assistant
```

Create a virtual environment

```bash
python -m venv venv
```

Activate it

### Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 API Key

Create a `.env` file

```
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
```

---

## ▶️ Run the Project

```bash
python app.py
```

---

## 🏗️ RAG Pipeline

```
PDF Documents
      │
      ▼
Extract Text
      │
      ▼
Chunking
      │
      ▼
Sentence Transformer Embeddings
      │
      ▼
ChromaDB
      │
      ▼
Similarity Search
      │
      ▼
Gemini LLM
      │
      ▼
Final Answer
```

---

## 📌 Example Questions

- What is the minimum attendance required?
- How can I apply for medical leave?
- What are the library rules?
- What are the placement eligibility criteria?
- What scholarships are available?

---

## 📈 Future Improvements

- Streamlit Web UI
- Upload custom PDFs
- Chat history
- Better metadata filtering
- Cloud deployment
- Authentication

---

## 👨‍💻 Author

**Vikas Gowda**

GitHub: https://github.com/VikasGowda-M

---

⭐ If you like this project, consider giving it a star!
