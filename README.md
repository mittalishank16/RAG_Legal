# LegalRAG

A Retrieval-Augmented Generation (RAG) system for legal document analysis using large language models, vector databases, and a conversational interface.

---

## Overview

LegalRAG is an AI-powered system designed to help legal professionals efficiently analyze and query legal documents. It enables users to upload legal files (such as PDFs and text documents), convert them into embeddings, and retrieve relevant information through a conversational interface.

The system follows a complete RAG pipeline:

- Document ingestion  
- Text chunking  
- Embedding generation  
- Vector storage  
- Context-aware answer generation using an LLM  

---

## Features

- Document upload and processing (supports `.pdf`, `.txt`, and `.docx`)
- Text preprocessing and segmentation
- Embedding generation using HuggingFace models
- Efficient document storage and retrieval using ChromaDB
- Semantic search for relevant legal context
- Natural language querying interface
- Context-aware answer generation using LLM (Llama via Groq)

---

## Architecture

```
User Query
    ↓
Retriever (Vector Database - ChromaDB)
    ↓
Relevant Document Chunks
    ↓
LLM (Groq - Llama)
    ↓
Final Answer (Context-Aware)
```

---

## Tech Stack

- LLM: Llama (via Groq Inference)
- Framework: LangChain
- Embeddings: HuggingFace
- Vector Database: ChromaDB
- Document Processing: PyMuPDF, TextLoader
- Environment Management: python-dotenv
- Language: Python

---

## Project Structure

```
.
├── data/                      # Input legal documents
├── ingestion_pipeline/        # Document processing and embedding logic
├── vectorstore/               # ChromaDB storage
├── rag_pipeline/              # Retrieval and generation logic
├── second_LegalRAG.ipynb      # End-to-end implementation notebook
└── README.md
```

---

## Pipeline Breakdown

### 1. Document Loading

- Loads documents from a specified directory
- Supports multiple file formats

### 2. Text Splitting

- Splits documents into smaller chunks
- Uses configurable parameters:
  - `chunk_size`
  - `chunk_overlap`

### 3. Embedding Generation

- Converts text chunks into vector embeddings using HuggingFace models

### 4. Vector Storage

- Stores embeddings in ChromaDB
- Enables efficient similarity-based retrieval

### 5. Retrieval and Generation

- Retrieves top-k relevant document chunks
- Passes retrieved context to the LLM
- Generates grounded, context-aware responses

---

## Example Usage

```python
query = "What are the key obligations in this contract?"
```

Output:

- A context-aware legal response based on retrieved document chunks

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-link>
cd LegalRAG
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_api_key_here
```

---

## Running the Project

Open Jupyter Notebook:

```bash
jupyter notebook
```

Run the notebook:

```
second_LegalRAG.ipynb
```

---

## Future Improvements

- Add FastAPI backend for API-based interaction  
- Integrate Pinecone for scalable vector storage  
- Build a Streamlit-based user interface  
- Implement legal-specific prompt engineering  
- Enable multi-document querying and summarization  

---

## Use Cases

- Legal research automation  
- Contract analysis  
- Case law retrieval  
- Internal knowledge search for law firms  

---

## Disclaimer

This project is intended for informational and assistance purposes only. It should not be used as a substitute for professional legal advice.