from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma

embedding_model="gemini-embedding-2-preview"
llm_model="gemini-2.5-flash"
# 1. Get Embedding Model
def get_embeddings_model()->GoogleGenerativeAIEmbeddings: 
      print("Initializing embedding model")
      try: 
         embedding_model=GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            api_key="AIzaSyAvZr7UIH3TA-5_cn8teArtErp140CsphU"
            )
         print(f"Initialized embedding model: {embedding_model}")
         return embedding_model
      
      except Exception as e:
         print(f"Error initializing embedding model: {e}")
         return None


#2. Get Vector Store
def get_vector_store(embedding_model)->Chroma:
    print(f"Initializing vector store with embedding model: {llm_model}")
    try: 
      vector_store=Chroma(
         collection_name="doc_chunks",
         embedding_function=embedding_model,
         persist_directory="./db/chroma_db",
         collection_metadata={"hnsw:space": "cosine"}
      )
      print("Initialized vector store")
      return vector_store
    
    except Exception as e:
      print(f"Error initializing vector store {e}")
      return None

def get_llm()->ChatGoogleGenerativeAI:
    print(f"Initializing LLM model: {llm_model}")
    try:
       llm=ChatGoogleGenerativeAI(
          model=llm_model,
          api_key="AIzaSyAvZr7UIH3TA-5_cn8teArtErp140CsphU",
          temperature=0.6
       )
       print(f"Initialized LLM model : {llm_model}")
       return llm
    except Exception as e:
       print(f"Error initializing llm model {e}")
       return None
    
