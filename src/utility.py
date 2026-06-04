import os
import json
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.stores import InMemoryStore
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

embedding_model="gemini-embedding-2-preview"
llm_model="llama-3.3-70b-versatile"

PARENT_CHUNKS_PATH = "./db/parent_chunks.json"

# 1. Get Embedding Model
def get_embeddings_model()->GoogleGenerativeAIEmbeddings:
      print("Initializing embedding model")
      try:
         embeddings=GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            api_key=os.getenv("GOOGLE_API_KEY")
            )
         print(f"Initialized embedding model: {embedding_model}")
         return embeddings

      except Exception as e:
         print(f"Error initializing embedding model: {e}")
         return None


#2. Get Vector Store
def get_vector_store(embedding_model, collection_name="doc_chunks")->Chroma:
    print(f"Initializing vector store with collection: {collection_name}")
    try:
      vector_store=Chroma(
         collection_name=collection_name,
         embedding_function=embedding_model,
         persist_directory="./db/chroma_db",
         collection_metadata={"hnsw:space": "cosine"}
      )
      print(f"Initialized vector store for collection: {collection_name}")
      return vector_store

    except Exception as e:
      print(f"Error initializing vector store {e}")
      return None

def get_llm()->ChatGroq:
    print(f"Initializing LLM model: {llm_model}")
    try:
       llm=ChatGroq(
          model=llm_model,
          api_key=os.getenv("GROQ_API_KEY"),
          temperature=0.6
       )
       print(f"Initialized LLM model : {llm_model}")
       return llm
    except Exception as e:
       print(f"Error initializing llm model {e}")
       return None


# 3. Parent Chunks Store helpers
def get_parent_chunks_store() -> InMemoryStore:
    """Load InMemoryStore from disk if it exists, otherwise return an empty store."""
    store = InMemoryStore()
    if os.path.exists(PARENT_CHUNKS_PATH):
        try:
            with open(PARENT_CHUNKS_PATH, "r") as f:
                data = json.load(f)
            pairs = [
                (item["id"], Document(page_content=item["page_content"], metadata=item["metadata"]))
                for item in data
            ]
            store.mset(pairs)
            print(f"Loaded {len(pairs)} parent chunks from {PARENT_CHUNKS_PATH}")
        except Exception as e:
            print(f"Error loading parent chunks store: {e}")
    return store


def save_parent_chunks_store(store: InMemoryStore) -> None:
    """Serialize InMemoryStore to disk."""
    try:
        os.makedirs(os.path.dirname(PARENT_CHUNKS_PATH), exist_ok=True)
        keys = list(store.yield_keys())
        docs = store.mget(keys)
        data = [
            {"id": k, "page_content": d.page_content, "metadata": d.metadata}
            for k, d in zip(keys, docs)
            if d is not None
        ]
        with open(PARENT_CHUNKS_PATH, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(data)} parent chunks to {PARENT_CHUNKS_PATH}")
    except Exception as e:
        print(f"Error saving parent chunks store: {e}")
