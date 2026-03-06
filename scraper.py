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
    """Fetches top N organic URLs from Google using SerpApi."""
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        print("Error: SERPAPI_API_KEY not found in .env file.")
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
        
        urls = [result.get("link") for result in organic_results[:num_results]]
        print(f"Found {len(urls)} URLs.")
        return urls
    except Exception as e:
        print(f"Error during Google Search: {e}")
        return []

def validate_and_save(url, markdown_content, index, data_dir):
    """Checks content quality and saves if valid."""
    if not markdown_content:
        return False

    # Validation criteria
    text_lower = markdown_content.lower()
    stop_phrases = [
        '404 error', 'access denied', 'just a moment', 
        'cloudflare', 'captcha', 'page not found'
    ]
    
    # 1. Length check (> 1000 symbols)
    if len(markdown_content) <= 1000:
        print(f"URL [{url}] skipped: failed quality check (too short)")
        return False

    # 2. Stop-phrases check
    if any(phrase in text_lower for phrase in stop_phrases):
        print(f"URL [{url}] skipped: failed quality check (stop-phrases found)")
        return False

    # Save to file
    filename = f"competitor_{index}.md"
    filepath = os.path.join(data_dir, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    
    print(f"Successfully saved to {filepath}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Scrape competitor data for SEO analysis.")
    parser.add_argument("--query", type=str, default="SEO продвижение 2026", help="Search query")
    parser.add_argument("--num", type=int, default=10, help="Number of URLs to scrape")
    args = parser.parse_args()

    firecrawl_key = os.getenv("FIRECRAWL_API_KEY")
    if not firecrawl_key:
        print("Error: FIRECRAWL_API_KEY not found in .env file.")
        return

    app = FirecrawlApp(api_key=firecrawl_key)

    # Dynamic search
    target_urls = get_top_urls(args.query, args.num)

    if not target_urls:
        print("No URLs to scrape. Exiting.")
        return

    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    else:
        # Clear old data files
        for f in os.listdir(data_dir):
            if f.startswith("competitor_") and f.endswith(".md"):
                os.remove(os.path.join(data_dir, f))

    print(f"Starting scraping of {len(target_urls)} URLs with validation...")
    
    valid_count = 0
    for i, url in enumerate(target_urls, 1):
        print(f"Scraping [{i}/{len(target_urls)}]: {url}...")
        
        try:
            # Scrape content as Markdown
            scrape_result = app.scrape(url, formats=['markdown'])
            markdown_content = getattr(scrape_result, 'markdown', None)
            
            if validate_and_save(url, markdown_content, i, data_dir):
                valid_count += 1
                
        except Exception as e:
            print(f"API Error for {url}: {e}")
        
        time.sleep(1)

    print(f"\nScraping complete. Successfully saved {valid_count} documents.")

if __name__ == "__main__":
    main()
