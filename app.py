import streamlit as st
import subprocess
import os
import sys

st.set_page_config(page_title="SEO Анализатор Конкурентов", page_icon="🚀", layout="wide")

st.title("🚀 SEO Анализатор Конкурентов (RAG)")
st.markdown("""
Этот инструмент анализирует ТОП-10 выдачи поисковых систем по вашему запросу, 
извлекает лучшие практики конкурентов и формирует экспертный SEO-отчет с помощью ИИ.
""")

st.sidebar.header("Настройки")
query = st.sidebar.text_input("Ключевой запрос:", placeholder="Например: SEO продвижение 2026")
num_competitors = st.sidebar.slider("Количество конкурентов:", 1, 10, 5)

if st.sidebar.button("Запустить анализ"):
    if not query:
        st.error("Пожалуйста, введите запрос!")
    else:
        try:
            with st.status("Процесс анализа запущен...", expanded=True) as status_bar:
                # Вспомогательная функция для запуска скриптов с захватом ошибок
                def run_step(script, args, step_name):
                    status_bar.write(step_name)
                    cmd = [sys.executable, script] + args
                    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
                    if result.returncode != 0:
                        st.error(f"Ошибка в {script}:\n{result.stderr}")
                        raise Exception(f"Step {script} failed")
                    return result

                # Шаг 1: Scraper
                run_step("scraper.py", ["--query", query, "--num", str(num_competitors)], "🔍 [1/4] Поиск в Google и парсинг сайтов...")
                
                # Шаг 2: Vectorizer
                run_step("vectorize.py", [], "🧠 [2/4] Создание векторов и загрузка в Pinecone...")
                
                # Шаг 3: Analyzer
                run_step("analyzer.py", [], "🤖 [3/4] Генерация отчета нейросетью...")
                
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
                st.error("Ошибка: файл финального отчета не найден.")
                
        except Exception as e:
            st.error(f"Произошла ошибка в процессе выполнения: {e}")

else:
    st.info("Введите запрос в боковой панели и нажмите 'Запустить анализ', чтобы начать.")
