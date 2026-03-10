import os
import sys
import io
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate

# 0. Фикс кодировки для логов
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 1. Загрузка переменных окружения
load_dotenv()

def main():
    print("🚀 Запуск analyzer.py (Gemini 3.1 Flash Lite)...")
    
    google_api_key = os.getenv("GOOGLE_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    if not google_api_key or not pinecone_api_key:
        print("❌ Error: Проверьте ключи GOOGLE_API_KEY и PINECONE_API_KEY!")
        return

    # 2. Инициализация Vector Store
    index_name = "seo-analysis"
    print(f"📡 Подключение к Pinecone: {index_name}...")
    
    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        openai_api_key=openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1"
    )
    
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )

    # 3. Инициализация LLM: Gemini 3.1 Flash Lite
    # Выбрана за высокие лимиты и скорость
    print("🧠 Использование модели: gemini-3.1-flash-lite")
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=google_api_key,
        temperature=0.1,
        max_output_tokens=8192
    )

    # 4. Агрессивный системный промпт (Фокус на 4-5 пункты)
    system_prompt = (
        "You are a Senior SEO Critic. Your goal is to find gaps in competitor strategies. "
        "CONCISE RULES:\n"
        "- Do NOT list more than 5 items in any sub-category. Use 'etc.' for the rest.\n"
        "- Spend 20% of effort on sections 1-3, and 80% on sections 4-5.\n\n"
        "STRICT STRUCTURE:\n"
        "1. Executive Summary: Market Leaders (Brief).\n"
        "2. Content Blueprint: H1-H3 structure (Consensus only).\n"
        "3. Semantic Highlights: Key LSI and Entities (Top 10 only).\n"
        "4. Conversion & E-E-A-T Stack: Specific trust signals found.\n"
        "5. STRATEGIC GAP ANALYSIS: Find 3 things competitors are MISSING. "
        "What unique content or tool can we build to be 10% better? (e.g., specific calculators, local reviews, etc.)\n\n"
        "Context:\n"
        "{context}"
    )

    # 5. Сборка контекста из Pinecone
    print("🔍 Поиск данных в Pinecone (MMR)...")
    retriever = vector_store.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 20, "fetch_k": 50}
    )
    
    search_query = "SEO structure, trust signals, LSI keywords, conversion elements, competitive gaps"
    docs = retriever.invoke(search_query)
    
    formatted_context = ""
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        formatted_context += f"\n--- Фрагмент {i+1} (Источник: {source}) ---\n"
        formatted_context += f"Текст: {doc.page_content}\n"

    # 6. Генерация отчета
    print("✍️ Формирование экспертного отчета...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    chain = prompt | llm
    
    try:
        response = chain.invoke({
            "context": formatted_context, 
            "input": "Perform a deep-dive SEO gap analysis based on the competitor data provided."
        })
        
        answer = response.content 
        
        os.makedirs("data", exist_ok=True)
        with open("data/raw_report.md", "w", encoding="utf-8") as f:
            f.write(answer)
            
        print("✅ Успех! Отчет в 'data/raw_report.md'.")
        
    except Exception as e:
        print(f"❌ Ошибка генерации: {e}")

if __name__ == "__main__":
    main()
