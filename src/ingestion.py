from numpy.testing import print_assert_equal
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import json
import os


load_dotenv()

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
   if not doc: 
      raise ValueError("No document provided")
   print(f"Chunking documents: {len(doc)} pages")
   
   text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " "]
   )
   try:
      chunks=text_splitter.split_documents(doc)
      print(f"Split into {len(chunks)} chunks of {type(chunks[0])}")
      for i,chunk in enumerate(chunks):
         print(f"Chunk {i+1}: \n",json.dumps(chunk.dict(), indent=2))
      return chunks
   except Exception as e:
      print(f"Error chunking document: {e}")
      return None

# 3. Generate Embeddings
def generate_embeddings(chunks:list)->list:   
   if not chunks:
        raise ValueError("No chunks provided")

   print(f"Generating embeddings for {len(chunks)} chunks")
   try:
        embedding_model=GoogleGenerativeAIEmbeddings(
          model="gemini-embedding-2-preview",
          api_key="AIzaSyAbEFu0vgnOXQeGtd--1IjrbKhXpwyeLC4"
        )      
   except Exception as e:
        print(f"Error loading  embedding model : {e}")
        return None 
   try: 
      embeddings=embedding_model.embed_documents([chunk.page_content for chunk in chunks])
      return embeddings
   except Exception as e:
      print(f"Error generating embeddings: {e}")
      return None     

#4. Store Embedding
def store_embeddings(chunks:list,embeddings:list)->None:
    if not chunks: 
      raise ValueError("No chunks provided")

    print(f"Storing {len(chunks)} embeddings in Chroma DB")
    try: 
      chromaStore=Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="doclens_ragbot",
        persist_directory="./db/chromadb"
      )
      print("Embeddings stored successfully")
    except Exception as e:
      print(f"Error storing embeddings: {e}")
      return None



if  __name__=="__main__":
    docs=load_file(file_path)
    chunk_doc(docs)