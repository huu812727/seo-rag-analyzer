import os
import sys
import io
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate

# 0. Фикс кодировки для корректного вывода в логах
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 1. Загрузка переменных окружения
load_dotenv()

def main():
    print("🚀 Запуск analyzer.py (Google Gemini Native Edition)...")
    
    # Получаем ключи
    google_api_key = os.getenv("GOOGLE_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY") # Нужен для эмбеддингов
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    if not google_api_key or not pinecone_api_key:
        print("Error: Не найден GOOGLE_API_KEY или PINECONE_API_KEY!")
        return

    # 2. Инициализация Vector Store
    # Мы оставляем OpenAIEmbeddings, чтобы не пересоздавать индекс в Pinecone
    index_name = "seo-analysis"
    print(f"📡 Подключение к Pinecone index '{index_name}'...")
    
    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        openai_api_key=openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1"
    )
    
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )

    # 3. Инициализация нативного Gemini 2.0 Flash-Lite
    # Это самая актуальная и быстрая модель для RAG задач на сегодня
    print("🧠 Инициализация LLM: gemini-2.0-flash-lite...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        google_api_key=google_api_key,
        temperature=0.1, # Минимальная температура для исключения "воды"
        max_output_tokens=4000
    )

    # 4. Настройка системного промпта (Твоя версия "Без воды")
    system_prompt = (
        "You are a Senior Data-Driven SEO Strategist. Your task is to analyze raw Markdown text scraped from TOP competitor websites and generate a highly specific, actionable SEO blueprint. "
        "CRITICAL CONSTRAINTS:\n"
        "- NO FLUFF: Do not use introductory or concluding remarks. Start strictly with the first heading.\n"
        "- ZERO HALLUCINATIONS: Base your analysis STRICTLY on the provided context.\n"
        "- EXTREME SPECIFICITY: Quote exact terms, LSI keywords, and unique features found in the competitor data.\n\n"
        "REPORT STRUCTURE (Use Markdown):\n\n"
        "1. Executive Summary: Market Reality. Identify exact content formats and competitor names.\n\n"
        "2. Content Architecture (The Blueprint): Propose a high-converting H1. Map out exact H2-H3 hierarchy based on competitor consensus.\n\n"
        "3. Semantic Entity Map & LSI: Extract hard list of mandatory entities explicitly present in the context.\n\n"
        "4. Commercial & UX Conversion Stack: Identify specific E-E-A-T signals and UX elements found in the data.\n\n"
        "5. Strategic Gap Analysis: Provide 2 highly actionable, non-obvious recommendations to outperform competitors.\n\n"
        "Context:\n"
        "{context}"
    )

    # 5. Сборка контекста из Pinecone (MMR поиск для разнообразия данных)
    print("🔍 Поиск релевантных данных в Pinecone...")
    retriever = vector_store.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 20, "fetch_k": 50}
    )
    
    search_query = "SEO structure, headings, pricing, commercial factors, trust signals, reviews, LSI keywords"
    docs = retriever.invoke(search_query)
    
    print(f"🛠 Обработка {len(docs)} фрагментов контекста...")
    formatted_context = ""
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        formatted_context += f"\n--- Фрагмент {i+1} (Источник: {source}) ---\n"
        formatted_context += f"Текст: {doc.page_content}\n"

    # 6. Запуск генерации
    print("✍️ Генерация финального отчета...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    chain = prompt | llm
    
    try:
        response = chain.invoke({
            "context": formatted_context, 
            "input": "Create a detailed expert SEO report based on the provided competitor context."
        })
        
        # У Gemini ответ лежит в .content
        answer = response.content 
        
        os.makedirs("data", exist_ok=True)
        with open("data/raw_report.md", "w", encoding="utf-8") as f:
            f.write(answer)
            
        print("✅ Успех! Отчет сохранен в 'data/raw_report.md'.")
        
    except Exception as e:
        print(f"❌ Ошибка при генерации: {e}")

if __name__ == "__main__":
    main()
