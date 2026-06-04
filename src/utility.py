import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

embedding_model="gemini-embedding-2-preview"
llm_model="llama-3.3-70b-versatile"

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
    
