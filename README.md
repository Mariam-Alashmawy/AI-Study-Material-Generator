# AI-Study-Material-Generator
An AI-powered RAG system to generate quizzes and flashcards from PDFs
# 🎓 AI Study Buddy: RAG-Powered Quiz & Flashcard Generator

An end-to-end AI application that uses **Retrieval-Augmented Generation (RAG)** to transform PDF documents into interactive study materials. Built with **Mistral-Nemo-Instruct**, **FastAPI**, and **Streamlit**.



## 🚀 Quick Start (Recommended)

Because this project uses the heavy **Mistral-Nemo-Instruct-2407** model (approx. 24GB), running it locally requires a high-end GPU. We recommend using the provided notebook on Google Colab or Kaggle.

1. **Open the Notebook**: 
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/AI-Study-Buddy-RAG/blob/main/notebooks/final-project1.ipynb)
2. **Set Hardware**: Ensure the environment is set to **GPU (T4 or higher)**.
3. **Configure**: Insert your `NGROK_TOKEN` when prompted in the code.
4. **Run All**: Execute all cells to launch the FastAPI backend and the Streamlit frontend.

---

## ✨ Features

* **PDF Intelligence**: Extracts and chunks text from uploaded PDFs using `PyPDF2` and `RecursiveCharacterTextSplitter`.
* **Vector Search**: Uses `FAISS` and `HuggingFaceEmbeddings` to find the most relevant context for questions.
* **Interactive Quizzes**: Generate 5 multiple-choice questions with **zero-default selection** logic (no pre-selected answers).
* **Smart Flashcards**: Automated creation of front/back study cards.
* **Ngrok Integration**: Securely tunnels the local Streamlit app to a public URL for easy access.

---

## 🛠️ Tech Stack

* **LLM**: Mistral-Nemo-Instruct-2407
* **Orchestration**: LangChain
* **Vector DB**: FAISS (Facebook AI Similarity Search)
* **Backend**: FastAPI & Uvicorn
* **Frontend**: Streamlit
* **Tunneling**: PyNgrok

---

## 📝 Setup for Local Development

If you have a GPU with 24GB+ VRAM, you can run this locally:

1. **Clone the repo**:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/AI-Study-Buddy-RAG.git](https://github.com/YOUR_USERNAME/AI-Study-Buddy-RAG.git)
   cd AI-Study-Buddy-RAG
