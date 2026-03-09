# Imports

import os
from dotenv import load_dotenv
from typing import TypedDict

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from sentence_transformers import CrossEncoder

# Load Environment
load_dotenv()

# Embeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Load Constitution Vector DB

DB_PATH = r"C:\Users\user\Desktop\RAG Projects\Legal RAG 1\vector_database"

constitution_db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings,
    collection_name="legal_knowledge"
)

print("Total constitution chunks:", constitution_db._collection.count())

temp_doc_db = None

# Document Chunking Function

def split_documents(docs, embeddings, debug=False):

    structure_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800,
        chunk_overlap=250,
        separators=[
            r"\n\s*ISSUE",
            r"\n\s*FACTS",
            r"\n\s*BACKGROUND",
            r"\n\s*ARGUMENT",
            r"\n\s*ANALYSIS",
            r"\n\s*HOLDING",
            r"\n\s*ORDER",
            r"\n\s*JUDGMENT",
            r"\n\s*RELIEF",
            r"\n\s*Article\s+\d+",
            r"\n\s*Section\s+\d+",
            r"\n\s*\d+\.",
            r"\n\s*\(\d+\)",
            "\n\n",
            "\n",
            " "
        ],
        is_separator_regex=True
    )

    structured_docs = structure_splitter.split_documents(docs)

    try:
        semantic_splitter = SemanticChunker(embeddings)
        semantic_docs = semantic_splitter.split_documents(structured_docs)
    except:
        semantic_docs = structured_docs

    final_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=250
    )

    final_docs = final_splitter.split_documents(semantic_docs)

    return final_docs

# Load Uploaded Document function

from langchain_community.vectorstores import Chroma

def load_uploaded_document(file_path):

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Use your existing splitting utility
    chunks = split_documents(docs, embeddings)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    print("Uploaded document indexed:", len(chunks))

    return vector_store

# Legal Knowledge Retrieval function

def retrieve_legal_docs(query, k=5):

    docs = constitution_db.similarity_search(
        f"{query} legal principle",
        k=k
    )

    return docs

# Cross Encoder Reranker

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Uploaded Document Retrieval function

def retrieve_document_docs(query, k=15):

    global temp_doc_db

    if temp_doc_db is None:
        return []

    docs = temp_doc_db.similarity_search(query, k=k)

    pairs = [[query, d.page_content] for d in docs]

    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    return [doc for doc,_ in ranked[:5]]

# LLM 

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# Graph State

from typing import TypedDict, Optional, Any

class LegalState(TypedDict):
    question: str
    strategy: str
    plan: Optional[str]

    legal_context: Optional[str]
    document_context: Optional[str]

    document_db: Optional[Any]  # vector store for uploaded PDF

    answer: Optional[str]


# Strategist Agent

def strategist_agent(state: LegalState):

    question = state["question"]

    prompt = f"""
You are a legal strategist.

Decide the best research strategy.

Possible strategies:
- legal (use Indian law knowledge base)
- document (use uploaded case document)
- both (use both sources)

Respond ONLY in this format:

Strategy: <legal | document | both>
Reason: <short explanation>

Question:
{question}
"""

    response = llm.invoke(prompt)
    output = response.content

    # Extract strategy safely
    strategy = "legal"

    if "both" in output.lower():
        strategy = "both"
    elif "document" in output.lower():
        strategy = "document"
    elif "legal" in output.lower():
        strategy = "legal"

    return {
        "strategy": strategy,
        "plan": output
    }

# Legal Research Agent

def legal_research_agent(state: LegalState):

    docs = retrieve_legal_docs(state["question"])

    context = "\n\n".join([d.page_content for d in docs])

    return {"legal_context": context}

# Document Research Agent

def document_research_agent(state: LegalState):

    docs = retrieve_document_docs(state["question"], k=20)

    if len(docs) == 0:
        return {"document_context": ""}

    context = "\n\n".join([d.page_content for d in docs])

    return {"document_context": context}

# Advisor Agent

def advisor_agent(state: LegalState):

    question = state["question"]

    legal_context = state.get("legal_context", "")
    document_context = state.get("document_context", "")

    prompt = f"""
You are an expert Indian legal advisor.

Answer the question using the provided context.

If the answer is not present in the context,
say the information is not available.

LEGAL CONTEXT:
{legal_context}

DOCUMENT CONTEXT:
{document_context}

QUESTION:
{question}

Provide a clear explanation.
"""

    response = llm.invoke(prompt)

    return {"answer": response.content}

from langgraph.graph import StateGraph, END

# Initialize the workflow
workflow = StateGraph(LegalState)

# Add Nodes
workflow.add_node("strategist", strategist_agent)
workflow.add_node("legal_research", legal_research_agent)
workflow.add_node("document_research", document_research_agent)
workflow.add_node("advisor", advisor_agent)

# Set the Entry Point
workflow.set_entry_point("strategist")

# Updated Routing Function for Parallel Execution
def route_research(state: LegalState):
    """
    Determines the next step(s) in the workflow. 
    Returning a list triggers parallel execution in LangGraph.
    """
    strategy = state.get("strategy", "").lower()

    if "both" in strategy:
        # This triggers BOTH nodes to run simultaneously
        return ["legal_research", "document_research"]
    
    if "document" in strategy or "uploaded" in strategy:
        return ["document_research"]
    
    # Default to legal research
    return ["legal_research"]

# Updated Conditional Routing
# We no longer need the mapping dictionary because the function 
# returns the exact node names to visit.
workflow.add_conditional_edges(
    "strategist",
    route_research
)

# Connect the research nodes to the final advisor
# Note: LangGraph automatically handles the 'join' if both were running in parallel
workflow.add_edge("legal_research", "advisor")
workflow.add_edge("document_research", "advisor")

# Final step
workflow.add_edge("advisor", END)

# Compile the graph
legal_graph = workflow.compile()

print("Graph successfully updated with parallel routing logic.")


# RAG Main Agent Function

def legal_agentic_rag(question):

    result = legal_graph.invoke({
        "question": question
    })

    return result["answer"]