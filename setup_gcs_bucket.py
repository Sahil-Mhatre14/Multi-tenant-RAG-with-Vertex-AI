"""
GCS Bucket Setup for SJSU RAG Knowledge Base
=============================================
Creates and configures the GCS bucket for storing scraped web content
and uploaded files before ingesting into RAG Engine.

Usage:
    python3 setup_gcs_bucket.py

What this does:
- Creates bucket: gs://sjsu-it-genai-poc-kb
- Sets up folder structure: /it/, /housing/, /bursar/
- Configures IAM permissions
- Tests upload/download to verify it works
"""

from google.cloud import storage

# ============================================================================
# CONFIGURATION
# ============================================================================
PROJECT_ID = "sjsu-it-genai-poc"
BUCKET_NAME = "sjsu-rag-it-genai-poc-kb"
LOCATION = "us-west1"  # Same region as your RAG Engine corpus

# Departments that will have folders in the bucket
DEPARTMENTS = ["it", "housing", "bursar"]

# ============================================================================
# Initialize
# ============================================================================
print("Initializing GCS client...")
storage_client = storage.Client(project=PROJECT_ID)
print(f"✓ Connected to project: {PROJECT_ID}\n")

# ============================================================================
# STEP 1: Create Bucket
# ============================================================================
print(f"Step 1: Creating bucket '{BUCKET_NAME}'...")

try:
    # Check if bucket already exists
    bucket = storage_client.bucket(BUCKET_NAME)
    if bucket.exists():
        print(f"✓ Bucket already exists: gs://{BUCKET_NAME}")
    else:
        # Create new bucket
        bucket = storage_client.create_bucket(
            BUCKET_NAME,
            location=LOCATION,
        )
        print(f"✓ Created bucket: gs://{BUCKET_NAME}")
        print(f"  Location: {LOCATION}")
except Exception as e:
    print(f"✗ Error creating bucket: {e}")
    print("\nPossible issues:")
    print("1. Bucket name already taken globally (try a different name)")
    print("2. Missing permissions (need roles/storage.admin)")
    print("3. Billing not enabled on project")
    exit(1)

print()

# ============================================================================
# STEP 2: Create Department Folders (placeholder files)
# ============================================================================
print("Step 2: Setting up department folder structure...")

for dept in DEPARTMENTS:
    folder_path = f"{dept}/.placeholder"
    blob = bucket.blob(folder_path)
    
    # Upload a small placeholder file to "create" the folder
    # (GCS doesn't have real folders, but this makes them visible in Console)
    blob.upload_from_string(
        f"This folder contains {dept} department knowledge base files.",
        content_type="text/plain"
    )
    print(f"  ✓ Created folder: gs://{BUCKET_NAME}/{dept}/")

print()

# ============================================================================
# STEP 3: Test Upload/Download
# ============================================================================
print("Step 3: Testing upload and download...")

test_content = "This is a test file to verify GCS access works correctly."
test_path = "/IT_VP.pdf"

try:
    # Upload test file
    blob = bucket.blob(test_path)
    blob.upload_from_string(test_content, content_type="text/plain")
    print(f"  ✓ Uploaded test file: gs://{BUCKET_NAME}/{test_path}")
    
    # Download test file
    downloaded_content = blob.download_as_text()
    if downloaded_content == test_content:
        print(f"  ✓ Download verified successfully")
    else:
        print(f"  ✗ Download content doesn't match!")
    
    # Clean up test file
    blob.delete()
    print(f"  ✓ Cleaned up test file")
    
except Exception as e:
    print(f"  ✗ Test failed: {e}")
    exit(1)

print()

# ============================================================================
# STEP 4: Display Bucket Info
# ============================================================================
print("Step 4: Bucket configuration summary...")
bucket.reload()

print(f"  Name: {bucket.name}")
print(f"  Location: {bucket.location}")
print(f"  Storage class: {bucket.storage_class}")
print(f"  Created: {bucket.time_created}")

print()

# ============================================================================
# STEP 5: List Current Contents
# ============================================================================
print("Step 5: Current bucket contents...")
blobs = list(storage_client.list_blobs(BUCKET_NAME))

if blobs:
    print(f"  Found {len(blobs)} files:")
    for blob in blobs:
        size_kb = blob.size / 1024 if blob.size else 0
        print(f"    - {blob.name} ({size_kb:.1f} KB)")
else:
    print("  Bucket is empty")

print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 70)
print("GCS BUCKET SETUP COMPLETE")
print("=" * 70)
print(f"Bucket: gs://{BUCKET_NAME}")
print(f"Region: {LOCATION}")
print(f"\nDepartment folders:")
for dept in DEPARTMENTS:
    print(f"  - gs://{BUCKET_NAME}/{dept}/")
print("\nNext steps:")
print("1. Use scraper.py to scrape URLs and save to this bucket")
print("2. Manually upload files via Console: https://console.cloud.google.com/storage")
print("3. Import from bucket to RAG corpus using rag.import_files()")
print("\nQuick commands:")
print(f"  # List bucket contents")
print(f"  gsutil ls gs://{BUCKET_NAME}/")
print(f"\n  # Upload a file manually")
print(f"  gsutil cp myfile.pdf gs://{BUCKET_NAME}/it/")
print(f"\n  # View in Console")
print(f"  https://console.cloud.google.com/storage/browser/{BUCKET_NAME}")
print("=" * 70)