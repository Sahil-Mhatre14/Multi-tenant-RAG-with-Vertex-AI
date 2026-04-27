"""
URL Ingestion Demo
==================
Demonstrates scraping URLs and adding them to the IT knowledge base.

Usage:
    python3 demo_url_ingestion.py

Prerequisites:
    1. Run setup_gcs_bucket.py first
    2. Have a corpus created (from setup_it_corpus.py)
    3. Update CORPUS_NAME below
"""

import vertexai
from scraper import ingest_url_to_corpus, ingest_multiple_urls

# ============================================================================
# CONFIGURATION
# ============================================================================
PROJECT_ID = "sjsu-it-genai-poc"
LOCATION = "us-west1"

# TODO: Update this with your actual corpus name from setup_it_corpus.py
CORPUS_NAME = "projects/925509787316/locations/us-west1/ragCorpora/4035225266123964416"

# Department ID (corresponds to GCS folder)
DEPT_ID = "it"

# ============================================================================
# Example URLs to ingest
# ============================================================================
# Replace these with actual SJSU IT help page URLs
EXAMPLE_URLS = [
    # Add real SJSU IT URLs here, for example:
    # "https://www.sjsu.edu/it/services/collaboration/email/index.php",
    # "https://www.sjsu.edu/it/services/network/vpn/index.php",
    # "https://www.sjsu.edu/it/services/accounts/password/index.php",
]

# ============================================================================
# Initialize
# ============================================================================
print("Initializing Vertex AI...")
vertexai.init(project=PROJECT_ID, location=LOCATION)
print(f"✓ Connected to project: {PROJECT_ID}\n")

# Validate corpus name is set
if CORPUS_NAME == "YOUR_CORPUS_NAME_HERE":
    print("=" * 70)
    print("⚠ ERROR: CORPUS_NAME not set")
    print("=" * 70)
    print("\nPlease update CORPUS_NAME in this script.")
    print("\nSteps:")
    print("1. Run setup_it_corpus.py if you haven't already")
    print("2. Copy the corpus name from the output")
    print("3. Paste it into the CORPUS_NAME variable at the top of this file")
    print("=" * 70 + "\n")
    exit(1)

# ============================================================================
# DEMO 1: Ingest a single URL
# ============================================================================
print("\n" + "=" * 70)
print("DEMO 1: Ingest a Single URL")
print("=" * 70)

# Option 1: User provides URL interactively
user_url = input("\nEnter a URL to scrape (or press Enter to skip): ").strip()

if user_url:
    result = ingest_url_to_corpus(
        url=user_url,
        corpus_name=CORPUS_NAME,
        dept_id=DEPT_ID
    )
    
    if result['success']:
        print(f"\n✓ Success! URL has been added to the knowledge base.")
        print(f"  GCS Location: {result['gcs_uri']}")
        print(f"  Wait 1-2 minutes for embedding, then test with qna.py\n")
    else:
        print(f"\n✗ Failed: {result['message']}\n")
else:
    print("Skipped single URL demo.\n")

# ============================================================================
# DEMO 2: Batch ingest multiple URLs
# ============================================================================
print("\n" + "=" * 70)
print("DEMO 2: Batch Ingest Multiple URLs")
print("=" * 70)

if EXAMPLE_URLS:
    print(f"\nIngesting {len(EXAMPLE_URLS)} URLs from EXAMPLE_URLS list...\n")
    
    batch_result = ingest_multiple_urls(
        urls=EXAMPLE_URLS,
        corpus_name=CORPUS_NAME,
        dept_id=DEPT_ID
    )
    
    print(f"\n✓ Batch complete:")
    print(f"  Successful: {len(batch_result['successful'])}")
    print(f"  Failed: {len(batch_result['failed'])}")
    
else:
    print("\nNo URLs in EXAMPLE_URLS list.")
    print("Add URLs to the EXAMPLE_URLS list at the top of this file to test batch ingestion.\n")

# ============================================================================
# Next Steps
# ============================================================================
print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)
print("\n1. Wait 1-2 minutes for embedding to complete")
print("2. Test your new content with qna.py:")
print("   python3 qna.py")
print("\n3. View your GCS bucket:")
print("   gsutil ls -r gs://sjsu-it-genai-poc-kb/")
print("\n4. View in GCS Console:")
print("   https://console.cloud.google.com/storage/browser/sjsu-it-genai-poc-kb")
print("\n5. Check corpus files:")
print("   Run this in Python:")
print(f"   from vertexai import rag")
print(f"   files = list(rag.list_files(corpus_name='{CORPUS_NAME}'))")
print(f"   for f in files: print(f.display_name)")
print("=" * 70 + "\n")