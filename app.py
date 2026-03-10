import streamlit as st
import os
import sys
import glob
import subprocess
import shutil
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="SEO Анализатор Конкурентов", layout="wide")

st.title("🚀 SEO Анализатор Конкурентов (RAG)")
st.markdown("""
Этот инструмент анализирует ТОП выдачи поисковых систем по вашему запросу (от 1 до 10 конкурентов), 
извлекает лучшие практики и формирует экспертный SEO-отчет с помощью ИИ.
""")

st.sidebar.header("Настройки")
query = st.sidebar.text_input("Ключевой запрос:", placeholder="Например: SEO продвижение 2026")
num_competitors = st.sidebar.slider("Количество конкурентов:", 1, 10, 5)

def clear_all_data():
    """Cleans up local data folder and Pinecone index."""
    st.write("🧹 Очистка старых данных...")
    
    # 1. Local cleanup (и гарантия того, что папка существует)
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True) # Создаем папку, если ее нет
    
    files = glob.glob(os.path.join(data_dir, "*.md"))
    for f in files:
        try:
            os.remove(f)
        except Exception as e:
            st.error(f"Не удалось удалить файл {f}: {e}")
            
    # 2. Pinecone cleanup с защитой от пустой базы
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if pinecone_api_key:
        try:
            pc = Pinecone(api_key=pinecone_api_key)
            index_name = "seo-analysis"
            if index_name in [idx.name for idx in pc.list_indexes()]:
                index = pc.Index(index_name)
                index.delete(delete_all=True)
                st.write("✨ Индекс Pinecone очищен.")
        except Exception as e:
            if "404" in str(e) or "Namespace not found" in str(e):
                st.write("✅ База уже пуста, очистка не требуется.")
            else:
                st.error(f"Ошибка при очистке Pinecone: {e}")
                
    st.write("✅ Подготовка завершена. Начинаем свежий анализ...")

if st.sidebar.button("Запустить анализ"):
    if not query:
        st.error("Пожалуйста, введите запрос!")
    else:
        try:
            with st.status("Процесс анализа запущен...", expanded=True) as status_bar:
                # Шаг 0: Очистка
                clear_all_data()
                
              # Умная функция с выводом логов
                def run_step(script, args, step_name):
                    status_bar.write(step_name)
                    cmd = [sys.executable, script] + args
                    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
                    
                    # ВЫВОДИМ ЛОГИ ПРЯМО В ИНТЕРФЕЙС
                    with status_bar.expander(f"Логи: {script}", expanded=False):
                        if result.stdout:
                            st.code(result.stdout, language="text")
                        if result.stderr:
                            st.error(result.stderr)
                            
                    # ЖЕСТКАЯ ПРОВЕРКА: ловим тихие ошибки (с поправкой на парсер)
                    is_fatal = False
                    if result.returncode != 0:
                        is_fatal = True
                    elif script == "scraper.py":
                        # Для парсера фатально только если он не скачал вообще ничего
                        if "Successfully saved 0 documents" in result.stdout or "Error: FIRECRAWL_API_KEY" in result.stdout:
                            is_fatal = True
                    else:
                        # Для остальных скриптов любое слово Error - это красный флаг
                        if "Error" in result.stdout or "Exception" in result.stdout:
                            is_fatal = True
                            
                    if is_fatal:
                        st.error(f"Скрытая ошибка в {script}! Раскройте спойлер 'Логи' выше, чтобы увидеть причину.")
                        raise Exception(f"Step {script} failed silently")
                    return result

                # Шаг 1: Scraper
                run_step("scraper.py", ["--query", query, "--num", str(num_competitors)], "🔍 [1/4] Поиск в Google и парсинг сайтов...")
                
                # Шаг 2: Vectorizer
                run_step("vectorize.py", [], "🧠 [2/4] Создание векторов и загрузка в Pinecone...")
                
                # Шаг 3: Analyzer
                run_step("analyzer.py", ["--query", query], "🤖 [3/4] Генерация отчета")
                
                # Шаг 4: Translator
                run_step("translator.py", [], "🇷🇺 [4/4] Перевод отчета на русский язык...")
                
                status_bar.update(label="✅ Анализ успешно завершен!", state="complete", expanded=False)

            # Вывод результата
            final_report_path = os.path.join("data", "final_report_ru.md")
            if os.path.exists(final_report_path):
                with open(final_report_path, "r", encoding="utf-8") as f:
                    report_content = f.read()
                    
                st.divider()
                st.markdown("### 📄 Финальный SEO-отчет")
                st.markdown(report_content)
                
                st.download_button(
                    label="📥 Скачать отчет (.md)",
                    data=report_content,
                    file_name=f"seo_analysis_{query.replace(' ', '_')}.md",
                    mime="text/markdown"
                )
            else:
                st.error("Ошибка: файл финального отчета не найден. Проверьте логи скрипта translator.py.")
                
        except Exception as e:
            st.error(f"Анализ остановлен.")

else:
    st.info("Введите запрос в боковой панели и нажмите 'Запустить анализ', чтобы начать.")
