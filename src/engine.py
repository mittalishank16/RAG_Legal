import os
import torch
from crewai import LLM, Agent, Task, Crew, Process
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from crewai.tools import tool

# 1. Secure API Key Handling
# When deployed, these will be pulled directly from the environment
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please set it in your environment/Secrets.")

# Internal CrewAI requirement: provide a dummy OpenAI key if using other providers
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "NA"

# 2. Initialize Models (Singleton Pattern)
device = "cuda" if torch.cuda.is_available() else "cpu"

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={'device': device}
)

db_path = "./vector_database"
vector_store = Chroma(persist_directory=db_path, embedding_function=embeddings)

# Initialize LLMs using the Groq API key via the environment
legal_llm = LLM(model="groq/llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
research_llm = LLM(model="groq/llama-3.1-8b-instant", api_key=GROQ_API_KEY)

# 3. Define the Custom RAG Tool
@tool("legal_research_tool")
def legal_research_tool(query: str):
    """
    Searches the official legal vector database. 
    Use this tool to find specific statutes, constitutional articles, and case law.
    """
    docs = vector_store.similarity_search(query, k=3)
    return "\n\n".join([f"Source Content: {doc.page_content}" for doc in docs])

# 4. Agent and Task Factory
def create_legal_crew():
    researcher = Agent(
        role='Legal Researcher',
        goal='Retrieve exact legal provisions and citations related to {user_query}',
        backstory='Expert in Indian Constitutional law and criminal statutes.',
        tools=[legal_research_tool],
        llm=research_llm,
        verbose=True,
        allow_delegation=False
    )

    strategist = Agent(
        role='Legal Strategist',
        goal='Analyze the research findings and determine legal hierarchy.',
        backstory='Senior legal counsel specializing in jurisdictional conflicts.',
        llm=legal_llm,
        verbose=True
    )

    advisor = Agent(
        role='Legal Advisor',
        goal='Synthesize a final, easy-to-understand response for the user.',
        backstory='Compassionate advisor who simplifies complex law.',
        llm=legal_llm,
        verbose=True
    )

    research_task = Task(
        description='Search for the top 2 relevant legal provisions for: {user_query}.',
        expected_output='A summary of the top 2 legal provisions in bullet points.',
        agent=researcher
    )

    strategy_task = Task(
        description='Analyze research findings and explain legal implications.',
        expected_output='A strategic analysis of the legal hierarchy.',
        agent=strategist
    )

    advisory_task = Task(
        description='Summarize the strategy into a professional final answer.',
        expected_output='A final response that directly answers the user query.',
        agent=advisor
    )

    return Crew(
        agents=[researcher, strategist, advisor],
        tasks=[research_task, strategy_task, advisory_task],
        process=Process.sequential,
        memory=False, 
        verbose=True,
        max_rpm=2    
    )

def run_legal_rag(user_query: str):
    crew = create_legal_crew()
    result = crew.kickoff(inputs={"user_query": user_query})
    return result.raw

if __name__ == "__main__":
    test_query = "What are the rights of a person under arrest?"
    print(run_legal_rag(test_query))