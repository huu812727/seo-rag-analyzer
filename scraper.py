import os
import time
import sys
import argparse
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from serpapi import GoogleSearch

# 1. Загрузка переменных
load_dotenv()

def get_top_urls(query, num_results=10):
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        print("Error: SERPAPI_API_KEY missing.")
        return []

    print(f"Searching Google for: '{query}' ({num_results} results)...")
    params = {
        "q": query,
        "location": "Moscow, Russia",
        "hl": "ru",
        "gl": "ru",
        "api_key": api_key,
        "num": num_results
    }

    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        return [result.get("link") for result in results.get("organic_results", [])[:num_results]]
    except Exception as e:
        print(f"Error during Google Search: {e}")
        return []

def validate_and_save(url, markdown_content, index, data_dir):
    if not markdown_content or len(markdown_content) <= 500: # Снизил порог до 500 для музыки/погоды
        print(f"URL [{url}] skipped: too short.")
        return False

    filename = f"competitor_{index}.md"
    filepath = os.path.join(data_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    print(f"Successfully saved to {filepath}")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="SEO продвижение 2026")
    parser.add_argument("--num", type=int, default=10)
    args = parser.parse_args()

    firecrawl_key = os.getenv("FIRECRAWL_API_KEY")
    app = FirecrawlApp(api_key=firecrawl_key)

    target_urls = get_top_urls(args.query, args.num)
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Очистка старых данных
    for f in os.listdir(data_dir):
        if f.startswith("competitor_"): os.remove(os.path.join(data_dir, f))

    valid_count = 0
    for i, url in enumerate(target_urls, 1):
        print(f"Scraping [{i}/{len(target_urls)}]: {url}...")
        try:
            # === УНИВЕРСАЛЬНЫЙ БЛОК ПАРСИНГА ===
            markdown_content = None
            
            # Попытка №1: Современный метод
            try:
                if hasattr(app, 'scrape_url'):
                    res = app.scrape_url(url, params={'formats': ['markdown'], 'onlyMainContent': True})
                    markdown_content = res.get('markdown') if isinstance(res, dict) else getattr(res, 'markdown', None)
            except:
                pass

            # Попытка №2: Если первый метод не сработал или его нет
            if not markdown_content:
                try:
                    # Старый метод (поддерживает formats как список или строку)
                    res = app.scrape(url, {'formats': ['markdown'], 'onlyMainContent': True})
                    markdown_content = res.get('markdown') if isinstance(res, dict) else getattr(res, 'markdown', None)
                except:
                    # Самый базовый вызов, если даже параметры не жрет
                    res = app.scrape(url)
                    markdown_content = res.get('markdown') if isinstance(res, dict) else getattr(res, 'markdown', None)

            if validate_and_save(url, markdown_content, i, data_dir):
                valid_count += 1
                
        except Exception as e:
            print(f"API Error for {url}: {e}")
        
        time.sleep(1)

    print(f"\nScraping complete. Successfully saved {valid_count} documents.")

if __name__ == "__main__":
    main()
