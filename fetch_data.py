import requests
import os
import time
from src.core.config import CORPUS_FILE

# List of 15 AI and Tech companies that are likely to have Vietnamese Wikipedia pages
companies = [
    "OpenAI",
    "DeepMind",
    "Microsoft",
    "Google",
    "Apple Inc.",
    "Meta Platforms",
    "Amazon.com",
    "Nvidia",
    "IBM",
    "Intel",
    "Samsung Electronics",
    "Tencent",
    "Alibaba Group",
    "Baidu",
    "FPT (công ty)"
]

HEADERS = {
    'User-Agent': 'GraphRAG-Lab-Project/1.0 (contact: admin@example.com)'
}

def fetch_wikipedia_content(title, lang="vi"):
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "redirects": 1
    }
    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if response.status_code != 200:
            return None
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if page_id == "-1":
                return None
            return page_data.get("extract", "")
    except:
        return None

def fetch_data():
    # Ensure data directory exists
    os.makedirs(os.path.dirname(CORPUS_FILE), exist_ok=True)
    
    print(f"Fetching Vietnamese Wikipedia pages for {len(companies)} companies...")
    corpus_text = ""
    
    for i, company in enumerate(companies):
        print(f"[{i+1}/{len(companies)}] Fetching {company}...")
        content = fetch_wikipedia_content(company)
        
        if not content:
            # Fallback search
            try:
                search_url = "https://vi.wikipedia.org/w/api.php"
                search_params = {"action": "opensearch", "format": "json", "search": company, "limit": 1}
                search_resp = requests.get(search_url, params=search_params, headers=HEADERS, timeout=10).json()
                if search_resp and len(search_resp) > 1 and search_resp[1]:
                    content = fetch_wikipedia_content(search_resp[1][0])
            except:
                pass

        if content:
            corpus_text += f"--- Thực thể: {company} ---\n{content[:5000]}\n\n"
        else:
            print(f"Could not find content for {company}")
        time.sleep(0.5)

    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        f.write(corpus_text)
    
    print(f"Hoàn thành! Đã lưu vào {CORPUS_FILE}")

if __name__ == "__main__":
    fetch_data()