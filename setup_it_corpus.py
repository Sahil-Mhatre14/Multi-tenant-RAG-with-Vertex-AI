"""
SJSU IT Service Desk - RAG Corpus Setup
========================================
This script:
1. Deletes the test corpus
2. Creates a new IT Service Desk corpus
3. Uploads PDF files to the corpus
4. Tests retrieval + generation with Gemini

Prerequisites:
- Run: gcloud auth application-default login
- Run: gcloud auth application-default set-quota-project sjsu-it-genai-poc
- Have some IT PDF files ready to upload
"""

import vertexai
from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool
import time

# ============================================================================
# CONFIGURATION
# ============================================================================
PROJECT_ID = "sjsu-it-genai-poc"
LOCATION = "us-west1"

# The test corpus you created earlier - update this ID
TEST_CORPUS_ID = "4611686018427387904"

# PDF files to upload - update these paths to your actual files
PDF_FILES = [
    "/Users/spartan/Desktop/RAG Engine/Information Technology.pdf",
    "/Users/spartan/Desktop/RAG Engine/IT_VP.pdf",
]

# ============================================================================
# STEP 1: Initialize Vertex AI
# ============================================================================
print("Initializing Vertex AI...")
vertexai.init(project=PROJECT_ID, location=LOCATION)
print(f"✓ Connected to project: {PROJECT_ID}, region: {LOCATION}\n")

# ============================================================================
# STEP 2: Delete test corpus
# ============================================================================
print("Step 1: Cleaning up test corpus...")
test_corpus_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora/{TEST_CORPUS_ID}"

try:
    rag.delete_corpus(name=test_corpus_name)
    print(f"✓ Deleted test corpus: {test_corpus_name}\n")
except Exception as e:
    print(f"⚠ Could not delete test corpus (might already be deleted): {e}\n")

# ============================================================================
# STEP 3: Create IT Service Desk corpus
# ============================================================================
print("Step 2: Creating IT Service Desk corpus...")

# Configure embedding model explicitly
embedding_config = rag.RagEmbeddingModelConfig(
    vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
        publisher_model="publishers/google/models/text-embedding-005"
    )
)

corpus = rag.create_corpus(
    display_name="sjsu-it-servicedesk",
    description="SJSU IT Service Desk knowledge base - student-facing help documentation",
    backend_config=rag.RagVectorDbConfig(
        rag_embedding_model_config=embedding_config
    ),
)

print(f"✓ Created corpus: {corpus.display_name}")
print(f"  Resource name: {corpus.name}")
print(f"  Embedding model: text-embedding-005")
print(f"  Vector DB: RagManagedDb (Spanner)\n")

# Save the corpus name for later use
CORPUS_NAME = corpus.name

# ============================================================================
# STEP 4: Upload PDF files
# ============================================================================
print("Step 3: Uploading PDF files...")

if not PDF_FILES:
    print("⚠ No PDF files specified. Update the PDF_FILES list with your file paths.")
    print("\nExample:")
    print('  PDF_FILES = [')
    print('      "/Users/yourusername/Downloads/password-reset.pdf",')
    print('      "/Users/yourusername/Downloads/wifi-guide.pdf",')
    print('  ]')
    print("\nSkipping upload step...\n")
else:
    uploaded_files = []
    for pdf_path in PDF_FILES:
        try:
            print(f"  Uploading: {pdf_path}")
            rag_file = rag.upload_file(
                corpus_name=CORPUS_NAME,
                path=pdf_path,
                display_name=pdf_path.split('/')[-1],  # Use filename as display name
                description=f"IT help documentation"
            )
            uploaded_files.append(rag_file)
            print(f"  ✓ Uploaded: {rag_file.display_name}")
        except Exception as e:
            print(f"  ✗ Failed to upload {pdf_path}: {e}")
    
    print(f"\n✓ Uploaded {len(uploaded_files)} files successfully\n")
    
    # Wait a moment for processing
    print("Waiting 30 seconds for embedding to complete...")
    time.sleep(30)

# ============================================================================
# STEP 5: List files in corpus
# ============================================================================
print("Step 4: Verifying uploaded files...")
files = list(rag.list_files(corpus_name=CORPUS_NAME))

if not files:
    print("  No files found in corpus yet.")
    print("  If you just uploaded, wait a minute and run this again.\n")
else:
    print(f"  Found {len(files)} files in corpus:")
    for f in files:
        status = f.file_status.state if hasattr(f.file_status, 'state') else 'UNKNOWN'
        print(f"    - {f.display_name} [{status}]")
    print()

# ============================================================================
# STEP 6: Test retrieval (only if files exist)
# ============================================================================
if files:
    print("Step 5: Testing retrieval...")
    test_query = "How do I reset my password?"
    
    try:
        response = rag.retrieval_query(
            rag_resources=[rag.RagResource(rag_corpus=CORPUS_NAME)],
            text=test_query,
            rag_retrieval_config=rag.RagRetrievalConfig(
                top_k=3,
                filter=rag.Filter(vector_distance_threshold=0.5)
            ),
        )
        
        print(f'  Query: "{test_query}"')
        print(f"  Retrieved {len(response.contexts.contexts)} chunks:\n")
        
        for idx, context in enumerate(response.contexts.contexts[:3], 1):
            print(f"  Chunk {idx}:")
            print(f"    Source: {context.source_uri}")
            print(f"    Distance: {context.distance:.3f}")
            print(f"    Text preview: {context.text[:200]}...")
            print()
    except Exception as e:
        print(f"  ✗ Retrieval failed: {e}\n")

# ============================================================================
# STEP 7: Test full RAG with Gemini (only if files exist)
# ============================================================================
if files:
    print("Step 6: Testing full RAG with Gemini...")
    
    try:
        # Create RAG retrieval tool
        rag_tool = Tool.from_retrieval(
            retrieval=rag.Retrieval(
                source=rag.VertexRagStore(
                    rag_resources=[rag.RagResource(rag_corpus=CORPUS_NAME)],
                    rag_retrieval_config=rag.RagRetrievalConfig(top_k=5),
                )
            )
        )
        
        # Create model with RAG tool
        model = GenerativeModel(
            model_name="gemini-2.5-flash",
            tools=[rag_tool]
        )
        
        test_question = "How do I connect to SJSU WiFi?"
        print(f'  Question: "{test_question}"\n')
        
        response = model.generate_content(test_question)
        print(f"  Gemini's answer:")
        print(f"  {response.text}\n")
        
    except Exception as e:
        print(f"  ✗ RAG generation failed: {e}\n")

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 70)
print("SETUP COMPLETE")
print("=" * 70)
print(f"Corpus name: {CORPUS_NAME}")
print(f"Files in corpus: {len(files)}")
print("\nNext steps:")
print("1. Add more PDF files by updating PDF_FILES and re-running this script")
print("2. Test more queries using the corpus name above")
print("3. Build the FastAPI backend to expose this via an API")
print("\nTo use this corpus in your code:")
print(f'  CORPUS_NAME = "{CORPUS_NAME}"')
print("=" * 70)