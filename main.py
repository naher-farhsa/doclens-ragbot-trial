from src.ingestion import run_ingestion_pipeline
from src.retrieval import run_retrieval_pipeline    


file_path="./doc/AGI_One_Page_Summary.pdf"
query="What are the challenges in AGI?"

# ==========================================
# INGESTION: Independent chunking from original document
# Pipeline 1: Original Doc → Token Chunks (independent)
# Pipeline 2: Original Doc → Hierarchical Chunks (independent)
# ==========================================
run_ingestion_pipeline(file_path)

# ==========================================
# RETRIEVAL: Independent retrieval from both chunk types
# Step 1: Token Chunks Retrieval (logged in terminal)
# Step 2: Hierarchical Chunks Retrieval (logged in terminal)
# ==========================================
retrieved_results=run_retrieval_pipeline(query)

print(f"\n{'=' * 60}")
print(f"FINAL RESULTS for query: '{query}'")
print(f"{'=' * 60}")

# Token chunks results
token_docs = retrieved_results.get("token_docs", [])
print(f"\n--- Token Chunks Results ({len(token_docs) if token_docs else 0} docs) ---")
if token_docs:
    for i, doc in enumerate(token_docs):
        print(f"Document {i+1}: \n", doc.page_content)

# Hierarchical chunks results
hierarchical_docs = retrieved_results.get("hierarchical_docs", [])
print(f"\n--- Hierarchical Chunks Results ({len(hierarchical_docs) if hierarchical_docs else 0} docs) ---")
if hierarchical_docs:
    for i, doc in enumerate(hierarchical_docs):
        print(f"Document {i+1}: \n", doc.page_content)