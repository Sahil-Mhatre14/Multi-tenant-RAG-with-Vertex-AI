"""
Web Scraper for SJSU RAG Knowledge Base
========================================
Scrapes content from URLs, cleans it, and saves to GCS for RAG ingestion.

Main functions:
- scrape_url(url) - Extract clean content from a web page
- save_to_gcs(content, bucket_name, gcs_path) - Upload content to GCS
- ingest_url_to_corpus(url, corpus_name, dept_id) - Full pipeline: scrape → GCS → RAG

Usage:
    from scraper import ingest_url_to_corpus
    ingest_url_to_corpus(
        url="https://www.sjsu.edu/it/services/password.html",
        corpus_name="projects/.../ragCorpora/123",
        dept_id="it"
    )
"""

import requests
from bs4 import BeautifulSoup
from google.cloud import storage
from vertexai.preview import rag
import hashlib
import re
from urllib.parse import urlparse
from datetime import datetime

try:
    from config import BUCKET_NAME, PROJECT_ID
except ImportError:
    BUCKET_NAME = "sjsu-rag-it-genai-poc-kb"
    PROJECT_ID = "sjsu-it-genai-poc"

# User agent for web requests (identifies your scraper)
USER_AGENT = "SJSU-IT-RAG-Bot/1.0 (Knowledge Base Indexer)"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_filename_from_url(url):
    """
    Generate a clean filename from a URL.
    Example: https://www.sjsu.edu/it/services/password.html 
         --> sjsu-edu-it-services-password.txt
    """
    parsed = urlparse(url)
    # Combine domain and path
    path_part = parsed.netloc + parsed.path
    # Remove special chars, replace with hyphens
    clean = re.sub(r'[^\w\-]', '-', path_part)
    # Remove consecutive hyphens
    clean = re.sub(r'-+', '-', clean)
    # Remove leading/trailing hyphens
    clean = clean.strip('-').lower()
    # Add timestamp to avoid collisions
    timestamp = datetime.now().strftime('%Y%m%d')
    return f"{clean}-{timestamp}.txt"

def url_to_hash(url):
    """Generate a short hash from URL for unique identification."""
    return hashlib.md5(url.encode()).hexdigest()[:12]

# ============================================================================
# CORE SCRAPER FUNCTIONS
# ============================================================================

def scrape_url(url, timeout=10):
    """
    Fetch and extract clean text content from a URL.
    
    Args:
        url: The web page URL to scrape
        timeout: Request timeout in seconds
        
    Returns:
        dict with:
          - 'content': Cleaned text content
          - 'title': Page title
          - 'url': Original URL
          - 'success': Boolean
          - 'error': Error message if failed
    """
    try:
        # Fetch the page
        headers = {'User-Agent': USER_AGENT}
        response = requests.get(url, timeout=timeout, headers=headers)
        response.raise_for_status()  # Raise exception for 4xx/5xx
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        # Extract title
        title = soup.find('title')
        title_text = title.get_text().strip() if title else 'Untitled'
        
        # Try to find main content area (common patterns)
        main_content = None
        for selector in ['main', 'article', '[role="main"]', '#content', '.content']:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        # If no main content found, use body
        if not main_content:
            main_content = soup.find('body')
        
        if not main_content:
            return {
                'success': False,
                'error': 'Could not find any content on page',
                'url': url
            }
        
        # Extract text
        text = main_content.get_text(separator='\n', strip=True)
        
        # Clean up whitespace
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]  # Remove empty lines
        clean_text = '\n\n'.join(lines)
        
        # Add metadata header
        content_with_metadata = f"""URL: {url}
Title: {title_text}
Scraped: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

{clean_text}
"""
        
        return {
            'success': True,
            'content': content_with_metadata,
            'title': title_text,
            'url': url,
            'word_count': len(clean_text.split())
        }
        
    except requests.exceptions.Timeout:
        return {'success': False, 'error': f'Request timed out after {timeout}s', 'url': url}
    except requests.exceptions.HTTPError as e:
        return {'success': False, 'error': f'HTTP {e.response.status_code}: {e}', 'url': url}
    except requests.exceptions.RequestException as e:
        return {'success': False, 'error': f'Request failed: {str(e)}', 'url': url}
    except Exception as e:
        return {'success': False, 'error': f'Scraping error: {str(e)}', 'url': url}

def save_to_gcs(content, bucket_name, gcs_path):
    """
    Save content to Google Cloud Storage.
    
    Args:
        content: Text content to save
        bucket_name: GCS bucket name (without gs://)
        gcs_path: Path within bucket (e.g., "it/file.txt")
        
    Returns:
        Full GCS URI (e.g., "gs://bucket-name/it/file.txt") on success,
        None on failure
    """
    try:
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        
        # Upload with proper content type
        blob.upload_from_string(
            content,
            content_type='text/plain; charset=utf-8'
        )
        
        gcs_uri = f"gs://{bucket_name}/{gcs_path}"
        return gcs_uri
        
    except Exception as e:
        print(f"✗ Failed to upload to GCS: {e}")
        return None

def import_from_gcs_to_corpus(gcs_uri, corpus_name, display_name=None):
    """
    Import a file from GCS into a RAG corpus.
    
    Args:
        gcs_uri: Full GCS path (e.g., "gs://bucket/path/file.txt")
        corpus_name: RAG corpus resource name
        display_name: Optional display name for the file
        
    Returns:
        RAG file resource on success, None on failure
    """
    try:
        response = rag.import_files(
            corpus_name=corpus_name,
            paths=[gcs_uri],
            chunk_size=256,
            chunk_overlap=50,
        )
        
        # The import is async, but we get back metadata
        return response
        
    except Exception as e:
        print(f"✗ Failed to import to corpus: {e}")
        return None

# ============================================================================
# HIGH-LEVEL PIPELINE
# ============================================================================

def ingest_url_to_corpus(url, corpus_name, dept_id):
    """
    Complete pipeline: Scrape URL → Save to GCS → Import to RAG corpus.
    
    Args:
        url: Web page URL to scrape
        corpus_name: RAG corpus resource name (e.g., "projects/.../ragCorpora/123")
        dept_id: Department ID (e.g., "it", "housing", "bursar")
        
    Returns:
        dict with:
          - 'success': Boolean
          - 'gcs_uri': GCS path where content was saved
          - 'message': Status message
          - 'scrape_result': Result from scrape_url()
    """
    print(f"\n{'='*70}")
    print(f"Ingesting URL: {url}")
    print(f"Department: {dept_id}")
    print(f"{'='*70}\n")
    
    # Step 1: Scrape the URL
    print("Step 1: Scraping URL...")
    scrape_result = scrape_url(url)
    
    if not scrape_result['success']:
        print(f"✗ Scraping failed: {scrape_result['error']}")
        return {
            'success': False,
            'message': f"Scraping failed: {scrape_result['error']}",
            'scrape_result': scrape_result
        }
    
    print(f"✓ Scraped successfully")
    print(f"  Title: {scrape_result['title']}")
    print(f"  Content length: {scrape_result['word_count']} words\n")
    
    # Step 2: Generate filename and GCS path
    filename = generate_filename_from_url(url)
    gcs_path = f"{dept_id}/{filename}"
    
    print("Step 2: Saving to GCS...")
    print(f"  Path: gs://{BUCKET_NAME}/{gcs_path}")
    
    # Step 3: Upload to GCS
    gcs_uri = save_to_gcs(scrape_result['content'], BUCKET_NAME, gcs_path)
    
    if not gcs_uri:
        return {
            'success': False,
            'message': 'Failed to upload to GCS',
            'scrape_result': scrape_result
        }
    
    print(f"✓ Saved to GCS: {gcs_uri}\n")
    
    # Step 4: Import to RAG corpus
    print("Step 3: Importing to RAG corpus...")
    print(f"  Corpus: {corpus_name}")
    
    import_result = import_from_gcs_to_corpus(
        gcs_uri=gcs_uri,
        corpus_name=corpus_name,
        display_name=scrape_result['title']
    )
    
    if not import_result:
        return {
            'success': False,
            'gcs_uri': gcs_uri,
            'message': 'Saved to GCS but failed to import to corpus',
            'scrape_result': scrape_result
        }
    
    print(f"✓ Import queued successfully")
    print(f"  Note: Embedding happens asynchronously. Check corpus in 1-2 minutes.\n")
    
    print(f"{'='*70}")
    print(f"SUCCESS: URL ingested to knowledge base")
    print(f"{'='*70}")
    print(f"Source URL: {url}")
    print(f"GCS Location: {gcs_uri}")
    print(f"Corpus: {corpus_name}")
    print(f"{'='*70}\n")
    
    return {
        'success': True,
        'gcs_uri': gcs_uri,
        'message': 'Successfully ingested URL',
        'scrape_result': scrape_result,
        'filename': filename
    }

# ============================================================================
# BATCH PROCESSING
# ============================================================================

def ingest_multiple_urls(urls, corpus_name, dept_id):
    """
    Ingest multiple URLs in batch.
    
    Args:
        urls: List of URLs to scrape
        corpus_name: RAG corpus resource name
        dept_id: Department ID
        
    Returns:
        dict with 'successful', 'failed', and 'results' lists
    """
    results = []
    successful = []
    failed = []
    
    print(f"\n{'='*70}")
    print(f"BATCH INGESTION: {len(urls)} URLs")
    print(f"{'='*70}\n")
    
    for idx, url in enumerate(urls, 1):
        print(f"[{idx}/{len(urls)}] Processing: {url}")
        
        result = ingest_url_to_corpus(url, corpus_name, dept_id)
        results.append(result)
        
        if result['success']:
            successful.append(url)
        else:
            failed.append({'url': url, 'error': result['message']})
        
        print()  # Blank line between URLs
    
    # Summary
    print(f"{'='*70}")
    print(f"BATCH COMPLETE")
    print(f"{'='*70}")
    print(f"Total: {len(urls)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print(f"\nFailed URLs:")
        for item in failed:
            print(f"  ✗ {item['url']}")
            print(f"    {item['error']}")
    
    print(f"{'='*70}\n")
    
    return {
        'successful': successful,
        'failed': failed,
        'results': results
    }