from dotenv import load_dotenv
import os
# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


FAISS_PATH = "faiss_index"

# 1. LOAD ENV
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

# 2. INITIALIZE SPLITTER & EMBEDDINGS (FIXED ORDER)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 3. LOAD OR CREATE FAISS

def load_or_create_vectorstore():
    if os.path.exists(FAISS_PATH):
        print("Loading existing FAISS index...")
        return FAISS.load_local(
            FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        print("No FAISS found yet.")
        return None


vectorstore = load_or_create_vectorstore()

# 4. ADD PDF FUNCTION

def add_new_pdf(file_path):
    global vectorstore

    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"PDF Loaded: {len(documents)} pages")

    docs = splitter.split_documents(documents)

    for doc in docs:
        doc.metadata["source"] = file_path

    if vectorstore is None:
        print("Creating new FAISS index...")
        vectorstore = FAISS.from_documents(docs, embeddings)
    else:
        print("Adding to existing FAISS...")
        vectorstore.add_documents(docs)

    vectorstore.save_local(FAISS_PATH)
    print(f" Added {file_path} successfully!")


# ADD YOUR PDF HERE (run once or whenever needed)
add_new_pdf(r"C:\Users\Owner\Rag\data\fundamental_rights.pdf")

# 5. CREATE RETRIEVER

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 6. LOAD LLM

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

# 7. PROMPT

prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

Answer the question directly in a clean and natural way.
Start immediately with the answer.

Rules:
- No headings like "Key Points", "Summary", etc.
- No introductory phrases
- Use bullet points only if useful
- Bold important terms naturally
- Keep it conversational but structured

Context:
{context}

Question: {question}

Answer:
""")

# 8. RAG CHAIN

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 9. CHAT LOOP

while True:
    query = input("\n Ask your question (or type 'exit'): ")

    if query.lower() == "exit":
        break

    response = rag_chain.invoke(query)
    print("\nAnswer:\n", response)

