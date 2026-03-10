import os
import sys
import io
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore # Исправленный импорт (уберет Warning)
from langchain_core.prompts import ChatPromptTemplate

# 0. Fix encoding for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 1. Load environment variables
load_dotenv()

def main():
    print("🚀 Запуск analyzer.py...")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not pinecone_api_key or not openrouter_api_key:
        print("Error: Required API keys not found in .env file.")
        return

    # 2. Initialize Embeddings and Vector Store
    index_name = "seo-analysis"
    print(f"Connecting to Pinecone index '{index_name}'...")
    
    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        openai_api_key=openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1"
    )
    
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )

    # 3. Initialize LLM (OpenRouter)
    print("Initializing LLM via OpenRouter (Gemini 3 Flash Preview)...")
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        model="meta-llama/llama-3.3-70b-instruct:free"
    )

   # 4. Setup RAG Chain
    system_prompt = (
        "You are a Senior Data-Driven SEO Strategist. Your task is to analyze raw Markdown text scraped from TOP competitor websites and generate a highly specific, actionable SEO blueprint. "
        "CRITICAL CONSTRAINTS:\n"
        "- NO FLUFF: Do not use introductory or concluding remarks (e.g., 'Here is your report', 'In conclusion'). Start strictly with the first heading.\n"
        "- ZERO HALLUCINATIONS: Base your analysis STRICTLY on the provided context. If a metric, tool, or entity is not in the text, DO NOT invent it.\n"
        "- EXTREME SPECIFICITY: Quote exact terms, LSI keywords, and unique features found in the competitor data. Never use generic placeholders like '[Brand Name]' or '[Industry Term]'.\n\n"
        "REPORT STRUCTURE (Use Markdown):\n\n"
        "1. Executive Summary: Market Reality. Identify the exact content format winning the SERP (e.g., aggregator, calculator, deep-dive guide). What is the exact user intent being satisfied? Mention specific competitor names found in the text.\n\n"
        "2. Content Architecture (The Blueprint): Propose a high-converting H1. Map out the exact H2-H3 hierarchy based on competitor consensus. Detail 2-3 highly specific, unique content blocks (with examples) that top leaders use to retain users.\n\n"
        "3. Semantic Entity Map & LSI: Extract a hard list of mandatory entities, technical jargon, and relational LSI keywords explicitly present in the context. Group them logically (e.g., Core, Commercial, Trust).\n\n"
        "4. Commercial & UX Conversion Stack: Identify specific E-E-A-T signals (e.g., specific licenses, author bios) and UX elements (e.g., dynamic tables, custom widgets) actively used by the scraped competitors.\n\n"
        "5. Strategic Gap Analysis: Identify what critical information or UX feature is missing across these specific competitors. Provide 2 highly actionable, non-obvious recommendations to make our page 10% better.\n\n"
        "Context:\n"
        "{context}"
    )

    # === Ручная сборка контекста с метаданными ===
    print("🔍 Извлечение данных из Pinecone (MMR)...")
    
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 20, "fetch_k": 50})
    search_query = "SEO structure, H1, H2, H3 headings, pricing, commercial factors, delivery, product features, reviews"
    
    docs = retriever.invoke(search_query)
    
    print("🛠 Формирование обогащенного контекста...")
    formatted_context = ""
    for i, doc in enumerate(docs):
        formatted_context += f"\n--- Фрагмент {i+1} ---\n"
        
        if "source" in doc.metadata:
            formatted_context += f"Источник: {doc.metadata['source']}\n"
        if "Header 1" in doc.metadata:
            formatted_context += f"H1: {doc.metadata['Header 1']}\n"
        if "Header 2" in doc.metadata:
            formatted_context += f"H2: {doc.metadata['Header 2']}\n"
        if "Header 3" in doc.metadata:
            formatted_context += f"H3: {doc.metadata['Header 3']}\n"
            
        formatted_context += f"Текст: {doc.page_content}\n"

    print("🧠 Генерация SEO-отчета через LLM...")
    
    chain = prompt | llm
    
    try:
        llm_task = "Analyze the competitor base and create a detailed expert report according to your instructions."
        response = chain.invoke({
            "context": formatted_context, 
            "input": llm_task
        })
        
        answer = response.content 
        
        os.makedirs("data", exist_ok=True)
        with open("data/raw_report.md", "w", encoding="utf-8") as f:
            f.write(answer)
        print("Raw report saved to 'data/raw_report.md'.")
        
        print("\n=== RAW ANALYSIS RESULT ===\n")
        print(answer)
        
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()
