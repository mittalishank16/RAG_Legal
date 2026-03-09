import os
import re
import torch
import chromadb
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker

def run_ingestion():
    # 1. Setup Environment
    load_dotenv(override=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Using device: {device}")

    # 2. Paths Configuration 
    # Using relative paths for deployment compatibility
    pdf_path = "./data/Indian_Constitution.pdf"
    db_path = "./vector_database"

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f" PDF not found at {pdf_path}")

    # 3. Initialize Embeddings (BGE-Base v1.5 as per your notebook)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={'device': device}
    )

    # 4. Load PDF
    print("⏳ Loading PDF...")
    loader = PyMuPDFLoader(pdf_path)
    vector_docs = loader.load()
    
    # 5. Segment and Clean Documents
    # Slicing based on your notebook's logic: Articles (3-283) and Schedules (283-381)
    articles_and_preamble = vector_docs[3:283]
    schedules = vector_docs[283:381]
    
    all_docs = articles_and_preamble + schedules

    cleaned_docs = []
    for doc in all_docs:
        # Clean Hindi/Non-ASCII noise and footer underscores
        raw_text = doc.page_content
        cleaned_text = "".join(i for i in raw_text if ord(i) < 128)
        cleaned_text = re.split(r'_{2,}', cleaned_text)[0].strip()
        
        # Update Metadata
        raw_page = doc.metadata.get("page", 0)
        doc.metadata["page_label"] = str(int(raw_page) + 1)
        doc.metadata["section_type"] = "Schedule" if raw_page >= 283 else "Article"
        doc.page_content = cleaned_text
        cleaned_docs.append(doc)

    # 6. Semantic Chunking
    print("🚀 Starting Semantic Chunking (this may take a minute)...")
    semantic_chunker = SemanticChunker(
        embeddings, 
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90
    )

    semantic_chunks = []
    for doc in cleaned_docs:
        chunks = semantic_chunker.split_documents([doc])
        for chunk in chunks:
            # Maintain Small-to-Big context
            chunk.metadata["parent_context"] = doc.page_content
            semantic_chunks.append(chunk)

    # 7. Persist to ChromaDB
    print(f"💾 Saving {len(semantic_chunks)} chunks to {db_path}...")
    client = chromadb.PersistentClient(path=db_path)
    
    vector_store = Chroma.from_documents(
        documents=semantic_chunks,
        embedding=embeddings,
        client=client,
        collection_name="legal_knowledge"
    )

    print(f"✅ Success! Total chunks in DB: {vector_store._collection.count()}")

if __name__ == "__main__":
    run_ingestion()