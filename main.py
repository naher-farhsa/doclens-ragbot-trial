from src.ingestion import run_ingestion_pipeline
from src.retrieval import run_retrieval_pipeline    


file_path="./doc/AGI_One_Page_Summary.pdf"
query="What are the challenges in AGI?"

run_ingestion_pipeline(file_path)

retrieved_docs=run_retrieval_pipeline(query)

print(f"Retrieved documents {retrieved_docs} for query '{query}':")

for i,doc in enumerate(retrieved_docs):
    print(f"Document {i+1}: \n",doc.page_content)