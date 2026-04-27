# Option A: Upload a local file directly
rag_file = rag.upload_file(
    corpus_name=corpus.name,
    path="/path/to/some-it-doc.pdf",
    display_name="IT Password Reset Guide",
)
print(f"Uploaded: {rag_file.name}")