import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

import sys
import io

# 0. Fix encoding for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 1. Load environment variables
load_dotenv()

def main():
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not pinecone_api_key or not openrouter_api_key:
        print("Error: Required API keys not found in .env file.")
        return

    # 2. Initialize Embeddings and Vector Store
    index_name = "seo-analysis"
    print(f"Connecting to Pinecone index '{index_name}'...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )

    # 3. Initialize LLM (OpenRouter)
    print("Initializing LLM via OpenRouter (Gemini 3 Flash Preview)...")
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        model="google/gemini-3-flash-preview"
    )

    # 4. Setup RAG Chain
    # System Prompt (Senior SEO Version - English)
    system_prompt = (
        "You are a Senior SEO Analyst. Analyze the provided competitor content from TOP-10 search results and create a detailed report. "
        "The report must be in English and follow this structure (use Markdown):\n\n"
        "1. Executive Summary: Why are they in the TOP? Describe the dominant pattern and average technical weight of the pages.\n\n"
        "2. Content Skeleton (The Perfect Blueprint): Propose the ideal H1. Break down the H2-H3 structure justified by competitor frequency. List unique blocks of leaders.\n\n"
        "3. Semantic Entity Map: Create a thematic cloud. List mandatory entities (LSI) and recommended inclusion percentage.\n\n"
        "4. Commercial & UX Stack: Specify mandatory elements for conversion (tables, quizzes, trust signals, E-E-A-T factors) that competitors have.\n\n"
        "5. Gap Analysis (Window of Opportunity): What are competitors missing and what is our growth point (how to make content 10% better).\n\n"
        "Context:\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Combined documents chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # Retriever (increased k for more context)
    retriever = vector_store.as_retriever(search_kwargs={"k": 40})
    
    # Retrieval chain
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # 5. Execute test query
    query = "Analyze the competitor base and create a detailed expert report according to your instructions."
    print(f"Executing query: '{query}'...")
    
    try:
        response = rag_chain.invoke({"input": query})
        answer = response["answer"]
        
        # 6. Save raw report
        os.makedirs("data", exist_ok=True)
        with open("data/raw_report.md", "w", encoding="utf-8") as f:
            f.write(answer)
        print("Raw report saved to 'data/raw_report.md'.")
        
        # 7. Output result
        print("\n=== RAW ANALYSIS RESULT (ENGLISH) ===\n")
        print(answer)
        
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()
