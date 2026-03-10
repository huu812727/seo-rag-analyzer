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
    print("🧠 Инициализация LLM: gemini-2.5-flash...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=google_api_key,
        temperature=0.1, # Минимальная температура для исключения "воды"
        max_output_tokens=8192
    )

    # 4. Настройка системного промпта (Твоя версия "Без воды")
    system_prompt = (
        "You are a Senior SEO Critic and Strategist. Your goal is NOT to praise competitors, but to find their weaknesses. "
        "STRICT TEMPLATE (Do not change header names):\n\n"
        "1. Executive Summary: Market Leaders & Intent.\n"
        "2. Content Blueprint: Exact H1-H3 hierarchy based on consensus.\n"
        "3. Semantic Map: Keywords and Entities.\n"
        "4. Conversion Stack: UX/UI and E-E-A-T elements.\n"
        "5. STRATEGIC GAP ANALYSIS (Growth Points): This is the most important section. Compare the competitors and find exactly what is missing. "
        "Identify 2-3 specific features, content types, or data points that NONE of the analyzed competitors are doing well. "
        "Provide a '10% Better' strategy: what unique value can we add to beat them?\n\n"
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
