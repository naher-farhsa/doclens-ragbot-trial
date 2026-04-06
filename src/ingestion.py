# --------IMPORTANT Follow up-------------
#  - Disected Document Structure
#  - Indexing and Types
#  - Document Storing in Vectorstore
# ----------------------------------------

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from  src.utility import get_embeddings_model,get_vector_store
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

# 2. Chunk Document
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


# 4. Run Ingestion
def run_ingestion_pipeline(file_path:str):

   if file_path:
      print(f"Starting ingestion pipeline for file: {file_path}")
      try:
         doc=load_file(file_path)
         if doc:
            chunks=chunk_doc(doc)
            if chunks:
               embedding_model=get_embeddings_model()
               if embedding_model:
                  vector_store=get_vector_store(embedding_model)
                  if vector_store:
                     store_embeddings(chunks,vector_store)
      except Exception as e:
         print(f"Error in ingestion pipeline: {e}")
       
   else:
      print("No file path provided for ingestion")






if  __name__=="__main__":
    
   run_ingestion_pipeline(file_path)