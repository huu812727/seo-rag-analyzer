import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import sys
import io

# Fix encoding for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 1. Load environment variables
load_dotenv()

def main():
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        print("Error: OPENROUTER_API_KEY not found in .env file.")
        return

    # 2. Read raw report
    raw_report_path = "data/raw_report.md"
    if not os.path.exists(raw_report_path):
        print(f"Error: Raw report file '{raw_report_path}' not found.")
        return

    with open(raw_report_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # 3. Initialize LLM (OpenRouter - Gemini 3 Flash Preview)
    print("Initializing Translator LLM via OpenRouter (Gemini 3 Flash Preview)...")
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        model="meta-llama/llama-3.3-70b-instruct:free"
    )

    # 4. Setup Translation Prompt
    system_prompt = (
        "Ты профессиональный SEO-локализатор. Твоя задача — перевести предоставленный Markdown-отчет на русский язык. "
        "Сохрани всю Markdown-разметку, таблицы и списки. Профессиональные термины (H1, LSI, Core Web Vitals, CTR, ROI, E-E-A-T) оставляй на английском."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Markdown Report to Translate:\n\n{report}"),
        ]
    )

    chain = prompt | llm

    # 5. Execute translation
    print("Translating report to Russian...")
    try:
        response = chain.invoke({"report": raw_text})
        translated_text = response.content
        
        # 6. Save final report
        final_report_path = "data/final_report_ru.md"
        with open(final_report_path, "w", encoding="utf-8") as f:
            f.write(translated_text)
        print(f"Final report saved to '{final_report_path}'.")

        # 7. Output Result
        print("\n=== FINAL RUSSIAN REPORT ===\n")
        print(translated_text)

    except Exception as e:
        print(f"Error during translation: {e}")

if __name__ == "__main__":
    main()
