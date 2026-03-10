import os
import sys
import io
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# 0. Фикс кодировки для корректного вывода в логах (Streamlit Cloud и Windows)
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 1. Загрузка переменных окружения
load_dotenv()

def main():
    print("🚀 Запуск translator.py (Google Gemini Native Edition)...")
    
    # Получаем ключ Google
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("Error: GOOGLE_API_KEY не найден в настройках!")
        return

    # 2. Чтение сырого отчета из analyzer.py
    raw_report_path = "data/raw_report.md"
    if not os.path.exists(raw_report_path):
        print(f"Error: Файл отчета '{raw_report_path}' не найден. Проверьте шаг анализа.")
        return

    with open(raw_report_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # 3. Инициализация нативного Gemini 2.5
    # Используем ту же модель, что и в анализаторе для консистентности и скорости
    print("🧠 Инициализация Translator LLM: gemini-3.1-flash-lite...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=google_api_key,
        temperature=0.1, # Низкая температура для максимально точного перевода
        max_output_tokens=8196
    )

    # 4. Настройка промпта перевода
    # Оставляем жесткие ограничения ZERO CHAT, чтобы файл не содержал мусора
    system_prompt = (
        "You are an expert IT and SEO localizer. Your ONLY task is to translate the provided English Markdown report into professional Russian. "
        "CRITICAL CONSTRAINTS:\n"
        "- ZERO CHAT: Output ONLY the translated Markdown. Do NOT include any introductory or concluding phrases.\n"
        "- FORMATTING: Preserve ALL Markdown syntax perfectly (headings, bolding, bullet points, tables).\n"
        "- TECHNICAL VOCABULARY: Keep professional SEO and Web3 terms in English (e.g., H1, H2, LSI, Core Web Vitals, CTR, ROI, E-E-A-T, Intent, Dwell time, P2P).\n"
        "- NO COMMENTARY: Do not add your own explanations or notes. Start your response directly with the first character of the translated report."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Markdown Report to Translate:\n\n{report}"),
    ])

    chain = prompt | llm

    # 5. Выполнение перевода
    print("⏳ Перевод отчета на русский язык...")
    try:
        response = chain.invoke({"report": raw_text})
        translated_text = response.content
        
        # 6. Сохранение финального результата
        os.makedirs("data", exist_ok=True)
        final_report_path = "data/final_report_ru.md"
        with open(final_report_path, "w", encoding="utf-8") as f:
            f.write(translated_text)
            
        print(f"✅ Успех! Финальный отчет сохранен: '{final_report_path}'.")

        # 7. Вывод результата в логи для контроля
        print("\n=== FINAL RUSSIAN REPORT ===\n")
        print(translated_text)

    except Exception as e:
        print(f"❌ Ошибка при переводе: {e}")

if __name__ == "__main__":
    main()
