import wikipediaapi
import json
from tqdm import tqdm
import time

def get_wiki_pages(categories=["Azərbaycan tarixi", "Azərbaycan mədəniyyəti", 
                             "Azərbaycan ədəbiyyatı", "Azərbaycan coğrafiyası"], 
                  min_length=500, max_pages=1000):
    """
    Recursively collect substantial Azerbaijani Wikipedia pages from multiple categories
    """
    wiki = wikipediaapi.Wikipedia(
        language='az',
        extract_format=wikipediaapi.ExtractFormat.WIKI,
        user_agent='AzGPTDataCollector/1.0'
    )
    
    collected_pages = {}
    visited_pages = set()
    
    def collect_pages(category_title):
        if len(collected_pages) >= max_pages:
            return
            
        category = wiki.page(f"Kateqoriya:{category_title}")
        if not category.exists():
            print(f"Category not found: {category_title}")
            return
            
        # First, process all articles in this category
        for member in category.categorymembers.values():
            if len(collected_pages) >= max_pages:
                return
                
            if member.title in visited_pages:
                continue
                
            visited_pages.add(member.title)
            
            # Skip if it's a category or template page
            if member.title.startswith('Kateqoriya:') or member.title.startswith('Şablon:'):
                continue
                
            # Skip if content is too short
            if len(member.text) < min_length:
                continue
                
            collected_pages[member.title] = {
                'title': member.title,
                'text': member.text,
                'url': member.fullurl,
                'length': len(member.text)
            }
            print(f"Collected: {member.title} ({len(member.text)} chars)")
            
            # Delay to avoid hitting API limits
            time.sleep(0.1)
        
        # Then process subcategories
        for subcategory in category.categorymembers.values():
            if subcategory.title.startswith('Kateqoriya:'):
                collect_pages(subcategory.title.replace('Kateqoriya:', ''))
                
    # Start collection from each category
    for category in categories:
        print(f"\nStarting collection from category: {category}")
        collect_pages(category)
    
    return collected_pages

def preprocess_text(text):
    """
    Enhanced text preprocessing for Azerbaijani text
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Add space after punctuation if missing
    for punct in '.!?،؛:()[]{}«»':
        text = text.replace(punct, punct + ' ')
    
    # Fix common OCR errors in Azerbaijani text
    replacements = {
        'i': 'ı',  # Replace dotted i with dotless ı where appropriate
        'І': 'I',
        '...': '…',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text

def save_dataset(pages, output_file='az_wiki_data.json'):
    """
    Save collected pages to a JSON file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(pages)} pages to {output_file}")

def main():
    # Collect pages with minimum length requirement
    print("Starting data collection...")
    pages = get_wiki_pages(min_length=500, max_pages=100)  # 500 chars minimum length
    
    # Preprocess and save
    print("\nPreprocessing and saving data...")
    for title in pages:
        pages[title]['text'] = preprocess_text(pages[title]['text'])
    
    save_dataset(pages)
    
    # Print statistics
    total_chars = sum(page['length'] for page in pages.values())
    if pages:
        print(f"\nCollection complete!")
        print(f"Total pages: {len(pages)}")
        print(f"Total characters: {total_chars}")
        print(f"Average page length: {total_chars / len(pages):.2f} characters")
        
        # Print some titles as examples
        print("\nSample of collected articles:")
        for title in list(pages.keys())[:5]:
            print(f"- {title} ({pages[title]['length']} chars)")

if __name__ == "__main__":
    main()