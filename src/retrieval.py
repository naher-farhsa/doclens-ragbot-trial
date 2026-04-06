# --------IMPORTANT Follow up-------------
#  - BM25 Retriever, Vector Retriever
# ----------------------------------------


from src.utility import get_embeddings_model,get_vector_store
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from  langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

# 1. Load Model and Vector Store

def load_model_vector_store():
    try:
        print(f"Loading embedding model and vector store")
        embedding_model=get_embeddings_model()
        if embedding_model:
            vector_store=get_vector_store(embedding_model)
            if vector_store:
                print("Successfully loaded embedding model and vector store")
                return embedding_model,vector_store
            else:
                print("Failed to load vector store")
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

# 3.2 Run Retrieval Pipeline
def run_retrieval_pipeline(query:str):
    embedding_model,vector_store=load_model_vector_store()
    if embedding_model and vector_store:
        retriever=get_hybrid_retriever(embedding_model,vector_store)
        if retriever:
            print("Retriever is ready to use")
            retrived_docs=run_retrieval(query,retriever)
            print("Retrieval process completed")
            return retrived_docs
        else:
            print("Failed to initialize retriever")
            return None
    else:
        print("Failed to load model or vector store")
        return None   

if __name__=="__main__":
    run_retrieval_pipeline("What are the challenges in AGI?")