# --------IMPORTANT Follow up-------------
#  - BM25 Retriever, Vector Retriever
# ----------------------------------------


from src.utility import get_embeddings_model, get_vector_store, get_parent_chunks_store
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

# 1. Load Model and Vector Store

def load_model_vector_store(collection_name="doc_chunks"):
    try:
        print(f"Loading embedding model and vector store for collection: {collection_name}")
        embedding_model=get_embeddings_model()
        if embedding_model:
            vector_store=get_vector_store(embedding_model, collection_name=collection_name)
            if vector_store:
                print(f"Successfully loaded embedding model and vector store for collection: {collection_name}")
                return embedding_model,vector_store
            else:
                print(f"Failed to load vector store for collection: {collection_name}")
                return None,None
    except Exception as e:
        print(f"Error loading model and vector store: {e}")
        return None,None

# 2. Get Retriever
def get_hybrid_retriever(embedding_model:GoogleGenerativeAIEmbeddings,vector_store:Chroma)->EnsembleRetriever:
    try:
        print("Initializing retriever")
        if vector_store:
            # vector retriever works on stored vector embeddings (here vector store : chroma)
            vector_retriever=vector_store.as_retriever(
                search_kwargs={"k":1, "score_threshold":0.7},
                search_type="similarity_score_threshold"
                )

            # bm25 retriever works on raw text and metadata from the vector store (here vector store : chroma)

            # all_docs is a dict with keys "documents" and "metadatas", each containing a list of texts and corresponding metadata
            all_docs=vector_store.get(include=["documents","metadatas"])

            #texts is a list of document texts, metas is a list of corresponding metadata dicts, we create Document objects for BM25 retriever
            texts,metas=all_docs["documents"],all_docs["metadatas"]
            documents=[Document(page_content=t,metadata=m) for t,m in zip(texts,metas)]

            bm25_retriever=BM25Retriever.from_documents(documents=documents, k=1)
            if vector_retriever and bm25_retriever:

                # Ensemble retriever combines both vector and BM25 retrievers, we can assign weights to each retriever based on their importance (here 0.7 for vector and 0.3 for BM25)
                ensemble_retriever=EnsembleRetriever(retrievers=[vector_retriever,bm25_retriever],weights=[0.7,0.3])

                print("Initialized ensemble retriever with vector and BM25 retrievers")
                return ensemble_retriever
            else:
                print("Failed to initialize individual retrievers")
                return None
        else:
            print("Vector store not available for retriever")
            return None
    except Exception as e:
        print(f"Error initializing retriever: {e}")
        return None


# 3.1 Run Retrieval

def run_retrieval(query:str,retriever:EnsembleRetriever)->list:
    try:
        print(f"Running retrieval for query: {query}")
        if retriever:
            retrived_docs=retriever.invoke(query)
        print(f"Retrieved {len(retrived_docs)} documents for query: {query}")
        return retrived_docs
    except Exception as e:
        print(f"Error running retrieval: {e}")
        return None


# ==========================================
# TOKEN CHUNKS RETRIEVAL
# ==========================================
def run_token_retrieval(query:str):
    """
    Independent retrieval from token-based chunks.
    Retrieves from 'token_chunks' collection.
    """
    print("\n" + "=" * 60)
    print("[TOKEN RETRIEVAL] Starting token chunks retrieval")
    print(f"[TOKEN RETRIEVAL] Query: {query}")
    print("=" * 60)

    try:
        embedding_model, vector_store = load_model_vector_store(collection_name="token_chunks")
        if embedding_model and vector_store:
            retriever = get_hybrid_retriever(embedding_model, vector_store)
            if retriever:
                print("[TOKEN RETRIEVAL] Retriever is ready, executing query...")
                retrieved_docs = run_retrieval(query, retriever)

                if retrieved_docs:
                    print(f"\n[TOKEN RETRIEVAL] === RESULTS ===")
                    print(f"[TOKEN RETRIEVAL] Retrieved {len(retrieved_docs)} documents from token chunks")
                    for i, doc in enumerate(retrieved_docs):
                        print(f"\n[TOKEN RETRIEVAL] --- Document {i+1} ---")
                        print(f"[TOKEN RETRIEVAL] Content: {doc.page_content[:200]}...")
                        print(f"[TOKEN RETRIEVAL] Metadata: {doc.metadata}")
                    print(f"[TOKEN RETRIEVAL] === END TOKEN RESULTS ===\n")
                else:
                    print("[TOKEN RETRIEVAL] No documents retrieved from token chunks")

                print("[TOKEN RETRIEVAL] Token chunks retrieval completed")
                return retrieved_docs
            else:
                print("[TOKEN RETRIEVAL] Failed to initialize retriever for token chunks")
                return None
        else:
            print("[TOKEN RETRIEVAL] Failed to load model or vector store for token chunks")
            return None
    except Exception as e:
        print(f"[TOKEN RETRIEVAL] Error in token retrieval: {e}")
        return None


# ==========================================
# HIERARCHICAL CHUNKS RETRIEVAL (ParentDocumentRetriever)
# ==========================================
def run_hierarchical_retrieval(query:str):
    """
    Retrieval using ParentDocumentRetriever:
    - Searches child chunks in 'hierarchical_chunks' vector store for the best semantic match.
    - Looks up the corresponding parent chunks from the persisted InMemoryStore.
    - Returns full parent chunks as context for generation.
    """
    print("\n" + "=" * 60)
    print("[HIERARCHICAL RETRIEVAL] Starting hierarchical retrieval (ParentDocumentRetriever)")
    print(f"[HIERARCHICAL RETRIEVAL] Query: {query}")
    print("=" * 60)

    try:
        embedding_model, vector_store = load_model_vector_store(collection_name="hierarchical_chunks")
        if embedding_model and vector_store:
            # Load persisted parent chunks store
            parent_store = get_parent_chunks_store()

            child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base",
                chunk_size=400,
                chunk_overlap=60,
            )

            retriever = ParentDocumentRetriever(
                vectorstore=vector_store,
                docstore=parent_store,
                child_splitter=child_splitter,
                id_key="parent_id",
            )

            print("[HIERARCHICAL RETRIEVAL] Retriever is ready, executing query...")
            retrieved_docs = retriever.invoke(query)

            if retrieved_docs:
                print(f"\n[HIERARCHICAL RETRIEVAL] === RESULTS ===")
                print(f"[HIERARCHICAL RETRIEVAL] Retrieved {len(retrieved_docs)} parent documents")
                for i, doc in enumerate(retrieved_docs):
                    print(f"\n[HIERARCHICAL RETRIEVAL] --- Parent Document {i+1} ---")
                    print(f"[HIERARCHICAL RETRIEVAL] Content ({len(doc.page_content)} chars): {doc.page_content[:200]}...")
                    print(f"[HIERARCHICAL RETRIEVAL] Metadata: {doc.metadata}")
                print(f"[HIERARCHICAL RETRIEVAL] === END HIERARCHICAL RESULTS ===\n")
            else:
                print("[HIERARCHICAL RETRIEVAL] No parent documents retrieved")

            print("[HIERARCHICAL RETRIEVAL] Hierarchical retrieval completed")
            return retrieved_docs
        else:
            print("[HIERARCHICAL RETRIEVAL] Failed to load model or vector store for hierarchical chunks")
            return None
    except Exception as e:
        print(f"[HIERARCHICAL RETRIEVAL] Error in hierarchical retrieval: {e}")
        return None


# 3.2 Run Retrieval Pipeline (Updated with both retrieval pipelines)
def run_retrieval_pipeline(query:str):
    """
    Runs retrieval from BOTH chunking strategies independently:
    1st: Token Chunks Retrieval (hybrid BM25 + vector)
    2nd: Hierarchical Chunks Retrieval (ParentDocumentRetriever — returns parent chunks)
    """
    print("\n" + "#" * 60)
    print("# RETRIEVAL PIPELINE - Independent Retrieval from Both Chunk Types")
    print(f"# Query: {query}")
    print("#" * 60)

    # ==========================================
    # STEP 1: TOKEN CHUNKS RETRIEVAL (First)
    # ==========================================
    print("\n" + "#" * 60)
    print("# STEP 1: TOKEN CHUNKS RETRIEVAL")
    print("#" * 60)
    token_docs = run_token_retrieval(query)

    # ==========================================
    # STEP 2: HIERARCHICAL CHUNKS RETRIEVAL (Second)
    # ==========================================
    print("\n" + "#" * 60)
    print("# STEP 2: HIERARCHICAL CHUNKS RETRIEVAL (ParentDocumentRetriever)")
    print("#" * 60)
    hierarchical_docs = run_hierarchical_retrieval(query)

    # ==========================================
    # RETRIEVAL SUMMARY
    # ==========================================
    print("\n" + "=" * 60)
    print("[RETRIEVAL SUMMARY]")
    print(f"  Token chunks retrieved: {len(token_docs) if token_docs else 0}")
    print(f"  Hierarchical parent docs retrieved: {len(hierarchical_docs) if hierarchical_docs else 0}")
    print("=" * 60)

    # Return both results
    return {
        "token_docs": token_docs,
        "hierarchical_docs": hierarchical_docs
    }

if __name__=="__main__":
    run_retrieval_pipeline("What are the challenges in AGI?")
