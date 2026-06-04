# --------IMPORTANT Follow up-------------
#  - Disected Document Structure
#  - Indexing and Types
#  - Document Storing in Vectorstore
# ----------------------------------------

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from  src.utility import get_embeddings_model,get_vector_store
from src.token_chunker import create_token_chunks
from src.hierarchical_chunker import create_hierarchical_chunks
from src.export_chunks import export_chunks_to_jsonl
from dotenv import load_dotenv
import json
import os


load_dotenv()

file_path="./doc/AGI_One_Page_Summary.pdf"

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
    Independent hierarchical chunking pipeline.
    Original Doc → Parent Chunks → Child Chunks → Store in 'hierarchical_chunks' collection
    """
    print("\n" + "=" * 60)
    print("[HIERARCHICAL INGESTION] Starting hierarchical ingestion pipeline")
    print("=" * 60)
    
    try:
        # Step 1: Create hierarchical chunks from ORIGINAL document
        print("[HIERARCHICAL INGESTION] Creating hierarchical chunks from original document...")
        parent_chunks, child_chunks = create_hierarchical_chunks(
            doc,
            parent_chunk_size=1800,
            parent_chunk_overlap=200,
            child_chunk_size=400,
            child_chunk_overlap=60,
        )
        
        if child_chunks:
            # Step 2: Store child chunks in separate collection (child chunks are used for retrieval)
            print(f"[HIERARCHICAL INGESTION] Storing {len(child_chunks)} child chunks in 'hierarchical_chunks' collection")
            hierarchical_vector_store = get_vector_store(embedding_model, collection_name="hierarchical_chunks")
            if hierarchical_vector_store:
                store_embeddings(child_chunks, hierarchical_vector_store)
                print(f"[HIERARCHICAL INGESTION] Successfully stored {len(child_chunks)} child chunks")
            
            # Step 3: Export chunks to JSONL
            export_chunks_to_jsonl(parent_chunks, "outputs/hierarchical_parent_chunks.jsonl")
            export_chunks_to_jsonl(child_chunks, "outputs/hierarchical_child_chunks.jsonl")
            print(f"[HIERARCHICAL INGESTION] Hierarchical ingestion pipeline completed")
            return parent_chunks, child_chunks
        else:
            print("[HIERARCHICAL INGESTION] No hierarchical chunks generated")
            return None, None
    
    except Exception as e:
        print(f"[HIERARCHICAL INGESTION] Error in hierarchical ingestion: {e}")
        return None, None


# 4. Run Ingestion (Updated with both chunking pipelines)
def run_ingestion_pipeline(file_path:str):

   if file_path:
      print(f"\nStarting ingestion pipeline for file: {file_path}")
      try:
         # Step 1: Load original document
         doc=load_file(file_path)
         if doc:
            embedding_model=get_embeddings_model()
            if embedding_model:
               
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
               # Original Doc → Hierarchical Chunks
               # ==========================================
               print("\n" + "#" * 60)
               print("# PIPELINE 2: HIERARCHICAL CHUNKING (Independent)")
               print("# Original Doc → Parent Chunks → Child Chunks → Store in 'hierarchical_chunks' collection")
               print("#" * 60)
               parent_chunks, child_chunks = run_hierarchical_ingestion(doc, embedding_model)
               
               # Summary
               print("\n" + "=" * 60)
               print("[INGESTION SUMMARY]")
               print(f"  Token chunks: {len(token_chunks) if token_chunks else 0}")
               print(f"  Parent chunks: {len(parent_chunks) if parent_chunks else 0}")
               print(f"  Child chunks: {len(child_chunks) if child_chunks else 0}")
               print(f"  Token chunks stored in collection: 'token_chunks'")
               print(f"  Hierarchical child chunks stored in collection: 'hierarchical_chunks'")
               print(f"  JSONL exports saved to: outputs/")
               print("=" * 60)

      except Exception as e:
         print(f"Error in ingestion pipeline: {e}")
       
   else:
      print("No file path provided for ingestion")





if  __name__=="__main__":
    
   run_ingestion_pipeline(file_path)