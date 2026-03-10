import os
import time
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from serpapi import GoogleSearch

import sys
import argparse

# 1. Load environment variables
load_dotenv()

def get_top_urls(query, num_results=10):
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        print("Error: SERPAPI_API_KEY not found.")
        return []

    print(f"Searching Google for: '{query}' ({num_results} results)...")
    params = {
        "q": query,
        "location": "Moscow, Russia",
        "hl": "ru",
        "gl": "ru",
        "google_domain": "google.ru",
        "api_key": api_key,
        "num": num_results
    }

    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        organic_results = results.get("organic_results", [])
        return [result.get("link") for result in organic_results[:num_results]]
    except Exception as e:
        print(f"Error during Google Search: {e}")
        return []

def validate_and_save(url, markdown_content, index, data_dir):
    if not markdown_content or len(markdown_content) <= 1000:
        print(f"URL [{url}] skipped: too short or empty.")
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
    if not firecrawl_key:
        print("Error: FIRECRAWL_API_KEY missing.")
        return

    app = FirecrawlApp(api_key=firecrawl_key)
    target_urls = get_top_urls(args.query, args.num)

    if not target_urls:
        print("No URLs to scrape.")
        return

    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    for f in os.listdir(data_dir):
        if f.startswith("competitor_"): os.remove(os.path.join(data_dir, f))

    print(f"Starting scraping of {len(target_urls)} URLs...")
    
    valid_count = 0
    for i, url in enumerate(target_urls, 1):
        print(f"Scraping [{i}/{len(target_urls)}]: {url}...")
        try:
            # ИСПОЛЬЗУЕМ scrape_url — это стандарт для актуального SDK
            scrape_result = app.scrape_url(url, {
                'formats': ['markdown'], 
                'onlyMainContent': True
            })
            
            # Получаем контент (в разных версиях может быть словарем или объектом)
            if isinstance(scrape_result, dict):
                markdown_content = scrape_result.get('markdown')
            else:
                markdown_content = getattr(scrape_result, 'markdown', None)
            
            if validate_and_save(url, markdown_content, i, data_dir):
                valid_count += 1
                
        except Exception as e:
            print(f"API Error for {url}: {e}")
        
        time.sleep(1)

    print(f"\nScraping complete. Successfully saved {valid_count} documents.")

if __name__ == "__main__":
    main()
