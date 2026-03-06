🚀 SEO Competitor Analyzer (RAG Pipeline)
Автоматизированный AI-пайплайн для глубокого анализа SEO-конкурентов. Инструмент парсит поисковую выдачу Google, извлекает контент коммерческих сайтов, векторизует его и генерирует экспертный SEO-отчет уровня Senior Analyst с помощью связки LangChain и LLM.

🧠 Архитектура и Стек технологий
Проект построен на архитектуре RAG (Retrieval-Augmented Generation) с полной миграцией вычислений в облако для обхода ограничений слабых серверов (Streamlit Cloud).

Интерфейс: Streamlit (с внедренной системой "Рентген" для перехвата silent-ошибок процессов).

Поиск (SERP): SerpApi (поиск ТОП органической выдачи Google).

Скрапинг: Firecrawl API (обход Cloudflare, рендеринг JS, конвертация HTML в чистый Markdown).

Векторизация (Embeddings): OpenAI text-embedding-3-small (размерность 1536) через OpenRouter API.

Векторная БД: Pinecone (Serverless AWS). Реализован паттерн "Чистой доски" (автоматическая очистка индекса перед каждой сессией во избежание загрязнения контекста).

LLM (Генерация): Google Gemini 2.5 Flash / 3 Flash (через OpenRouter API) под управлением LangChain.

🛠 Ключевые инженерные решения
Cloud-to-Cloud Embeddings: Отказ от локальной модели sentence-transformers в пользу OpenAI API сократил время создания векторов на 80% и снизил нагрузку на CPU сервера.

Идемпотентность БД: Внедрена автоматическая проверка размерности индекса Pinecone. Если база не совпадает с моделью (например, 384 vs 1536), скрипт автоматически пересоздает пространство имен.

Smart Scraper Validation: Парсер автоматически отсеивает слишком короткие страницы-заглушки (менее 1000 символов), сохраняя токены LLM и чистоту базы знаний.

Subprocess Logging: Все этапы (scraper -> vectorize -> analyzer -> translator) запускаются изолированно с жестким контролем stderr и выводом логов прямо в UI интерфейса.

⚙️ Установка и локальный запуск
Клонируйте репозиторий:

Bash
git clone https://github.com/вашт-логин/seo-rag-analyzer.git
cd seo-rag-analyzer
Установите зависимости:

Bash
pip install -r requirements.txt
Создайте файл .env в корневой папке и добавьте ваши API-ключи:

Фрагмент кода
SERPAPI_API_KEY=your_serpapi_key
FIRECRAWL_API_KEY=your_firecrawl_key
PINECONE_API_KEY=your_pinecone_key
OPENROUTER_API_KEY=your_openrouter_key
Запустите приложение:

Bash
streamlit run app.py
📄 Структура генерируемого отчета
LLM запрограммирована выдавать стандартизированный Markdown-документ, включающий:

Executive Summary: Анализ доминирующего паттерна ранжирования ниши.

Content Skeleton: Идеальная структура H1-H3 на основе частотности блоков у конкурентов.

Semantic Entity Map: LSI-ядро и рекомендуемая плотность сущностей.

Commercial & UX Stack: Анализ E-E-A-T факторов и элементов конверсии (калькуляторы, таблицы).

Gap Analysis (Окно возможностей): Точки роста и уникальные блоки, которых нет у конкурентов в выдаче.
