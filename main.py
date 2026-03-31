from src.ingestion import load_doc,chunk_doc


file_path="./doc/AGI_One_Page_Summary.pdf"

docs=load_file(file_path)

chunk_doc(docs)