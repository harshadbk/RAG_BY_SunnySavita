import os
import weaviate
from dotenv import load_dotenv

# LangChain (stable)
from langchain_community.vectorstores import Weaviate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.prompts import PromptTemplate
from langchain_groq import ChatGroq

# ---------------- ENV ----------------
load_dotenv()

WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_CLUSTER")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---------------- WEAVIATE ----------------
auth_config = weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY)

client = weaviate.Client(
    url=WEAVIATE_URL,
    auth_client_secret=auth_config,
    timeout_config=(15, 120),
)

# ---------------- EMBEDDINGS ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------- LOAD DATA ----------------
folder_path = r"C:\Users\Owner\Rag\data"
all_docs = []

for file in os.listdir(folder_path):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(folder_path, file))
        pages = loader.load()

        for p in pages:
            p.metadata["source"] = file

        all_docs.extend(pages)

print("Data Loaded")

# ---------------- SPLIT ----------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)

docs = splitter.split_documents(all_docs)

docs = docs[:10]

# ---------------- MEMORY ----------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# ---------------- VECTOR STORE ----------------
vector_db = Weaviate.from_documents(
    docs,
    embeddings,
    client=client,
    by_text=False
)

print("Data stored")

retriever = vector_db.as_retriever(search_kwargs={"k": 4})

# ---------------- LLM ----------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"
)

# ---------------- CUSTOM PROMPT ----------------
prompt = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""
You are an expert AI assistant for Indian Constitution.

Use the context to answer clearly.

If answer not in context → 

Give:
- Clear explanation
- Key points (if needed)

Context:
{context}

Chat History:
{chat_history}

Question:
{question}

Answer:
"""
)

# ---------------- RAG CHAIN ----------------
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt},
    return_source_documents=True
)

# ---------------- CHAT LOOP ----------------
while True:
    query = input("\nAsk (or exit): ")

    if query.lower() == "exit":
        break

    result = rag_chain.invoke({"question": query})

    print("\nAnswer:\n", result["answer"])