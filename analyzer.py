import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import Pinecone as PineconeVectorStore
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
    print("🚀 Запуск analyzer.py...")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not pinecone_api_key or not openrouter_api_key:
        print("Error: Required API keys not found in .env file.")
        return

    # 2. Initialize Embeddings and Vector Store
    index_name = "seo-analysis"
    print(f"Connecting to Pinecone index '{index_name}'...")
    
    # === ГЛАВНОЕ ИСПРАВЛЕНИЕ: Синхронизируем размерность с Pinecone (1536) ===
    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        openai_api_key=openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1"
    )
    
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )

    # 3. Initialize LLM (OpenRouter) - Твоя верная модель
    print("Initializing LLM via OpenRouter (Gemini 3 Flash Preview)...")
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        model="google/gemini-3-flash-preview"
    )

    # 4. Setup RAG Chain
    system_prompt = (
        "You are a Senior SEO Analyst. Analyze the provided competitor content from provided search results and create a detailed report. "
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

    # === НОВАЯ ЛОГИКА: Ручная сборка контекста с метаданными ===
    print("🔍 Извлечение данных из Pinecone (MMR)...")
    
    # 1. Настраиваем ретривер (с твоими правками MMR)
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 20, "fetch_k": 50})
    
    # Твой новый, правильный запрос для поиска в БД
    search_query = "SEO structure, H1, H2, H3 headings, pricing, commercial factors, delivery, product features, reviews"
    
    # 2. Получаем сырые документы из базы (без участия LLM)
    docs = retriever.invoke(search_query)
    
    print("🛠 Формирование обогащенного контекста...")
    # 3. Собираем контекст руками, добавляя заголовки к тексту
    formatted_context = ""
    for i, doc in enumerate(docs):
        formatted_context += f"\n--- Фрагмент {i+1} ---\n"
        
        # Безопасно проверяем наличие ключей в словаре metadata
        if "source" in doc.metadata:
            formatted_context += f"Источник: {doc.metadata['source']}\n"
        if "Header 1" in doc.metadata:
            formatted_context += f"H1: {doc.metadata['Header 1']}\n"
        if "Header 2" in doc.metadata:
            formatted_context += f"H2: {doc.metadata['Header 2']}\n"
        if "Header 3" in doc.metadata:
            formatted_context += f"H3: {doc.metadata['Header 3']}\n"
            
        formatted_context += f"Текст: {doc.page_content}\n"

    # 4. Собираем LCEL-цепочку (LangChain Expression Language)
    print("🧠 Генерация SEO-отчета через LLM...")
    
    # Мы передаем prompt напрямую в llm (prompt | llm)
    chain = prompt | llm
    
    try:
        # Передаем наш собранный контекст и изначальную задачу для LLM
        # Заметь: input - это команда для LLM, а search_query был командой для базы данных
        llm_task = "Analyze the competitor base and create a detailed expert report according to your instructions."
        response = chain.invoke({
            "context": formatted_context, 
            "input": llm_task
        })
        
        # В новом синтаксисе результат лежит в .content
        answer = response.content 
        
        # 6. Save raw report
        os.makedirs("data", exist_ok=True)
        with open("data/raw_report.md", "w", encoding="utf-8") as f:
            f.write(answer)
        print("Raw report saved to 'data/raw_report.md'.")
        
        print("\n=== RAW ANALYSIS RESULT ===\n")
        print(answer)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
    
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
