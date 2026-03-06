import subprocess
import sys
import os
import io

# 0. Fix encoding for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def run_script(script_name, description):
    print(f"\n{'='*50}")
    print(f">>> {description}...")
    print(f"{'='*50}\n")
    
    try:
        # Using sys.executable to ensure we use the same Python environment
        result = subprocess.run([sys.executable, script_name], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[CRITICAL ERROR] Script '{script_name}' failed with return code {e.returncode}.")
        print("Pipeline aborted.")
        return False
    except Exception as e:
        print(f"\n[UNEXPECTED ERROR] {e}")
        return False

def main():
    print("\n🚀 Инициализация полноцикловой RAG-системы для SEO-анализа\n")
    
    steps = [
        ("scraper.py", "[1/4] Поиск в Google через SerpApi и парсинг конкурентов через Firecrawl"),
        ("vectorize.py", "[2/4] Векторизация данных и загрузка в Pinecone"),
        ("analyzer.py", "[3/4] RAG-аналитика контента и генерация сырого SEO-отчета (English)"),
        ("translator.py", "[4/4] Профессиональная локализация и перевод отчета на русский язык")
    ]
    
    for script, desc in steps:
        if not run_script(script, desc):
            sys.exit(1)
            
    print(f"\n{'='*50}")
    print("✅ ВСЕ ЭТАПЫ УСПЕШНО ЗАВЕРШЕНЫ")
    print("Финальный отчет доступен в папке: data/final_report_ru.md")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()
