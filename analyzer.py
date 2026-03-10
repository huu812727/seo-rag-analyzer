import os
import sys
import argparse
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate

# Загрузка переменных окружения
load_dotenv()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()

    print(f"🚀 Запуск анализа по запросу: {args.query}")
    
    google_api_key = os.getenv("GOOGLE_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    if not google_api_key or not openrouter_api_key:
        print("❌ ОШИБКА: Проверьте API ключи в окружении!")
        return

    # 1. Подключение к вектору
    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        openai_api_key=openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1"
    )
    
    vector_store = PineconeVectorStore(index_name="seo-analysis", embedding=embeddings)

    # 2. Инициализация LLM (Стабильный Gemini 1.5 Flash)
    print(f"🧠 Использование модели: gemini-1.5-flash")
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        google_api_key=google_api_key,
        temperature=0.1,
        max_output_tokens=8192
    )

    # 3. Поиск контекста в базе
    print(f"🔍 Ищу фрагменты в базе по теме: {args.query}...")
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 15})
    docs = retriever.invoke(args.query)
    
    if not docs:
        print("❌ ОШИБКА: В базе Pinecone не найдено ничего релевантного!")
        return

    print(f"✅ Найдено {len(docs)} фрагментов. Формирую отчет...")
    
    formatted_context = ""
    for i, doc in enumerate(docs):
        formatted_context += f"\n--- Фрагмент {i+1} ---\n{doc.page_content}\n"

    # 4. Жесткий промпт (Фокус на 5 раздел)
    system_prompt = (
        "You are a Senior SEO Critic. Your goal is NOT to describe competitors, but to find their WEAKNESSES. "
        "STRICT CONSTRAINTS:\n"
        "- Section 1-4: MAX 3 bullet points each. Be extremely brief.\n"
        "- Section 5 (STRATEGIC GAP ANALYSIS): Spend 70% of your response here. "
        "Find exactly 3 things that NONE of these competitors are doing. "
        "Provide a specific '10% Better' strategy for each gap.\n\n"
        "STRICT STRUCTURE:\n"
        "1. Executive Summary (Brief)\n"
        "2. Consensus Content Map\n"
        "3. Semantic & LSI Entities\n"
        "4. Conversion/Trust Stack\n"
        "5. STRATEGIC GAP ANALYSIS & GROWTH POINTS (The Meat)"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "User Query: {query}\n\nContext from competitors:\n{context}")
    ])
    
    chain = prompt | llm
    
    # 5. Генерация и сохранение с защитой от ошибок парсинга ответа
    try:
        response = chain.invoke({"query": args.query, "context": formatted_context})
        
        # Бронебойный извлекатель текста
        answer = response.content
        if isinstance(answer, list):
            text_parts = []
            for item in answer:
                if isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])
                else:
                    text_parts.append(str(item))
            answer = "".join(text_parts)
        else:
            answer = str(answer)

        if not answer.strip():
            print("❌ ОШИБКА: Модель вернула пустой ответ!")
            return

        os.makedirs("data", exist_ok=True)
        with open("data/raw_report.md", "w", encoding="utf-8") as f:
            f.write(answer)
            
        print("✅ Отчет успешно сохранен.")
        
    except Exception as e:
        print(f"❌ Ошибка LLM: {e}")

if __name__ == "__main__":
    main()
