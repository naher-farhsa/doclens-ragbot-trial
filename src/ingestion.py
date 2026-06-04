# --------IMPORTANT Follow up-------------
#  - Disected Document Structure
#  - Indexing and Types
#  - Document Storing in Vectorstore
# ----------------------------------------

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.stores import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever
from src.utility import get_embeddings_model, get_vector_store, save_parent_chunks_store, PARENT_CHUNKS_PATH
from src.token_chunker import create_token_chunks
from src.export_chunks import export_chunks_to_jsonl
from langchain_core.documents import Document
from dotenv import load_dotenv
import json
import os


load_dotenv()

file_path="./doc/IAI.pdf"

# 1. Load File
def load_file(file_path:str)->list:
    if not file_path:
       raise FileNotFoundError("No file path provided")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"Loading document: {file_path}")
    try:
      doc=PyPDFLoader(file_path).load()
      print(f"Loaded {len(doc)} pages , {type(doc)}")
      return doc
    except Exception as e:
      print(f"Error loading document: {e}")
      return None

# 2. Chunk Document (Original character-based chunking - kept for reference)
def chunk_doc(doc:list)->list:
   if doc:
      print(f"Chunking documents: {len(doc)} pages")
      try:
         text_splitter=RecursiveCharacterTextSplitter(
         chunk_size=1000,
         chunk_overlap=200,
         separators=["\n\n", "\n", ".", " "]
         )
         chunks=text_splitter.split_documents(doc)
         print(f"Split into {len(chunks)} chunks of {type(chunks[0])}")
         for i,chunk in enumerate(chunks):
            print(f"Chunk {i+1}: \n",json.dumps(chunk.dict(), indent=2))
         return chunks

      except Exception as e:
         print(f"Error chunking document: {e}")
         return None
   else:
     raise ValueError("No document provided")



# 3. Store Embeddings
def store_embeddings(chunks:list,vector_store:Chroma)->None:
   if chunks:
      print(f"Storing {len(chunks)} chunks in vector store")
      try:
         for i,chunk in enumerate(chunks):
            # Note: The add_documents method is used to add documents to an existing vector store instance,
            # while the from_documents class method is used to create a new vector store instance and populate it with documents in one step.
            vector_store.add_documents(documents=[chunk])
            print(f"Stored chunk {i+1}/{len(chunks)} in vector store")
      except Exception as e:
         print(f"Error storing embeddings: {e}")


# ==========================================
# TOKEN-BASED CHUNKING INGESTION PIPELINE
# ==========================================
def run_token_ingestion(doc:list, embedding_model):
    """
    Independent token-based chunking pipeline.
    Original Doc → Token Chunks → Store in 'token_chunks' collection
    """
    print("\n" + "=" * 60)
    print("[TOKEN INGESTION] Starting token-based ingestion pipeline")
    print("=" * 60)

    try:
        # Step 1: Create token chunks from ORIGINAL document
        print("[TOKEN INGESTION] Creating token chunks from original document...")
        token_chunks = create_token_chunks(
            doc,
            chunk_size=500,
            chunk_overlap=80,
        )

        if token_chunks:
            # Step 2: Store token chunks in separate collection
            print(f"[TOKEN INGESTION] Storing {len(token_chunks)} token chunks in 'token_chunks' collection")
            token_vector_store = get_vector_store(embedding_model, collection_name="token_chunks")
            if token_vector_store:
                store_embeddings(token_chunks, token_vector_store)
                print(f"[TOKEN INGESTION] Successfully stored {len(token_chunks)} token chunks")

            # Step 3: Export token chunks to JSONL
            export_chunks_to_jsonl(token_chunks, "outputs/token_chunks.jsonl")
            print(f"[TOKEN INGESTION] Token ingestion pipeline completed")
            return token_chunks
        else:
            print("[TOKEN INGESTION] No token chunks generated")
            return None

    except Exception as e:
        print(f"[TOKEN INGESTION] Error in token ingestion: {e}")
        return None


# ==========================================
# HIERARCHICAL CHUNKING INGESTION PIPELINE
# ==========================================
def run_hierarchical_ingestion(doc:list, embedding_model):
    """
    Hierarchical parent-child ingestion using ParentDocumentRetriever.
    - Child chunks are stored in 'hierarchical_chunks' vector store (for similarity search).
    - Parent chunks are stored in InMemoryStore persisted to db/parent_chunks.json (for context retrieval).
    """
    print("\n" + "=" * 60)
    print("[HIERARCHICAL INGESTION] Starting hierarchical ingestion pipeline")
    print("=" * 60)

    try:
        # Step 1: Define splitters
        parent_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=1800,
            chunk_overlap=200,
        )
        child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=400,
            chunk_overlap=60,
        )

        # Step 2: Clear existing hierarchical collection and recreate clean
        print("[HIERARCHICAL INGESTION] Clearing existing 'hierarchical_chunks' collection...")
        stale_store = get_vector_store(embedding_model, collection_name="hierarchical_chunks")
        if stale_store:
            stale_store.delete_collection()
            print("[HIERARCHICAL INGESTION] Cleared existing collection")
        hierarchical_vector_store = get_vector_store(embedding_model, collection_name="hierarchical_chunks")

        # Step 3: Initialize empty parent docstore
        parent_store = InMemoryStore()

        # Step 4: Initialize ParentDocumentRetriever
        # id_key="parent_id" means child chunks stored in vectorstore will have
        # metadata["parent_id"] = the key used to look up the parent in docstore
        retriever = ParentDocumentRetriever(
            vectorstore=hierarchical_vector_store, # for child chunks
            docstore=parent_store,                 # for parent chunks
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            id_key="parent_id",
        )

        # Step 5: Add original documents — retriever handles all splitting and storage
        print("[HIERARCHICAL INGESTION] Adding documents to ParentDocumentRetriever...")
        retriever.add_documents(doc) 

        child_count = hierarchical_vector_store._collection.count()
        parent_keys = list(parent_store.yield_keys())
        print(f"[HIERARCHICAL INGESTION] Stored {len(parent_keys)} parent chunks in docstore")
        print(f"[HIERARCHICAL INGESTION] Stored {child_count} child chunks in vector store")

        # Step 6: Persist parent store to disk
        save_parent_chunks_store(parent_store)

        # Step 7: Reconstruct chunk lists for JSONL export
        parent_docs = [d for d in parent_store.mget(parent_keys) if d is not None]

        all_children = hierarchical_vector_store.get(include=["documents", "metadatas"])
        child_docs = [
            Document(page_content=t, metadata=m)
            for t, m in zip(all_children["documents"], all_children["metadatas"])
        ]

        export_chunks_to_jsonl(parent_docs, "outputs/hierarchical_parent_chunks.jsonl")
        export_chunks_to_jsonl(child_docs, "outputs/hierarchical_child_chunks.jsonl")

        print(f"[HIERARCHICAL INGESTION] Hierarchical ingestion pipeline completed")
        return parent_docs, child_docs

    except Exception as e:
        print(f"[HIERARCHICAL INGESTION] Error in hierarchical ingestion: {e}")
        return None, None


# 3.1 Check if already ingested
def check_already_ingested(embedding_model) -> bool:
    try:
        token_store = get_vector_store(embedding_model, collection_name="token_chunks")
        hier_store = get_vector_store(embedding_model, collection_name="hierarchical_chunks")

        token_count = token_store._collection.count() if token_store else 0
        hier_count = hier_store._collection.count() if hier_store else 0
        parent_store_exists = os.path.exists(PARENT_CHUNKS_PATH) and os.path.getsize(PARENT_CHUNKS_PATH) > 2

        print(f"[CHECK INGESTION] 'token_chunks' collection contains {token_count} documents.")
        print(f"[CHECK INGESTION] 'hierarchical_chunks' collection contains {hier_count} documents.")
        print(f"[CHECK INGESTION] Parent chunks store exists: {parent_store_exists}")

        return token_count > 0 and hier_count > 0 and parent_store_exists
    except Exception as e:
        print(f"[CHECK INGESTION] Error checking ingestion status: {e}")
        return False


# 4. Run Ingestion (Updated with both chunking pipelines)
def run_ingestion_pipeline(file_path:str):

   if file_path:
      try:
         embedding_model=get_embeddings_model()
         if embedding_model:
            if check_already_ingested(embedding_model):
               print("\n" + "=" * 60)
               print("[INGESTION] Both token and hierarchical collections are already populated. Skipping ingestion.")
               print("=" * 60)
               return

            print(f"\nStarting ingestion pipeline for file: {file_path}")
            # Step 1: Load original document
            doc=load_file(file_path)
            if doc:
               # ==========================================
               # PIPELINE 1: Token-based chunking (INDEPENDENT)
               # Original Doc → Token Chunks
               # ==========================================
               print("\n" + "#" * 60)
               print("# PIPELINE 1: TOKEN-BASED CHUNKING (Independent)")
               print("# Original Doc → Token Chunks → Store in 'token_chunks' collection")
               print("#" * 60)
               token_chunks = run_token_ingestion(doc, embedding_model)

               # ==========================================
               # PIPELINE 2: Hierarchical chunking (INDEPENDENT)
               # Original Doc → Hierarchical Chunks via ParentDocumentRetriever
               # ==========================================
               print("\n" + "#" * 60)
               print("# PIPELINE 2: HIERARCHICAL CHUNKING (ParentDocumentRetriever)")
               print("# Original Doc → Parent Chunks (docstore) + Child Chunks (vectorstore)")
               print("#" * 60)
               parent_chunks, child_chunks = run_hierarchical_ingestion(doc, embedding_model)

               # Summary
               print("\n" + "=" * 60)
               print("[INGESTION SUMMARY]")
               print(f"  Token chunks: {len(token_chunks) if token_chunks else 0}")
               print(f"  Parent chunks: {len(parent_chunks) if parent_chunks else 0}")
               print(f"  Child chunks: {len(child_chunks) if child_chunks else 0}")
               print(f"  Token chunks stored in collection: 'token_chunks'")
               print(f"  Child chunks stored in collection: 'hierarchical_chunks'")
               print(f"  Parent chunks persisted to: {PARENT_CHUNKS_PATH}")
               print(f"  JSONL exports saved to: outputs/")
               print("=" * 60)

      except Exception as e:
         print(f"Error in ingestion pipeline: {e}")

   else:
      print("No file path provided for ingestion")


if  __name__=="__main__":

   run_ingestion_pipeline(file_path)
