# test_imports.py
from langchain_huggingface import HuggingFaceEmbeddings
print("Import OK")

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Embeddings OK")