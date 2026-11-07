import os
import re # It is a powerful tool for searching, matching, and extracting specific text patterns using "regex.
import glob #glob lets you use a wildcard (*) to say: "Hey Python, just look inside the folder and grab every file that ends with .vtt

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv() 

DATA_DIR = "data"
DB_DIR = "chroma_store"


# 1. LOAD ---- read each transcript, throw away the VTT timestamps
def load_transcripts():
    docs = []
    for path in glob.glob(f"{DATA_DIR}/*.vtt"):
        lines = []
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line or line == "WEBVTT" or "-->" in line:
                continue
            lines.append(line)
        text = " ".join(lines)

        session = re.search(r"Session[ _]*(\d+)", path).group(1) # automatically pulls out just the session number (e.g., 1) so it can store it as metadata inside your vector database.

        docs.append(Document(page_content=text, metadata={"session": session}))

    return docs


# 2. BUILD ---- chunk, embed locally, and keep it on disk
def load_store():
    # Use Chroma's default local sentence-transformer model (all-MiniLM-L6-v2)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(DB_DIR):
        print("Loading existing vector store from disk...")
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    print("Building vector store locally (no API limits)...")
    docs = load_transcripts()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    ).split_documents(docs)

    return Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)


def build_retriever():
    return load_store().as_retriever(search_kwargs={"k": 5})


# 3. TRY IT ---- python src/retriever.py
if __name__ == "__main__":
    retriever = build_retriever()

    query = "what is regression testing?"
    print(f"\nSearching for: '{query}'...\n")
    results = retriever.invoke(query)
    
    print(f"=== Found {len(results)} Results ===\n")
    for i, r in enumerate(results, 1):
        print(f"Result {i} [Session {r.metadata['session']}]:")
        print(f"{r.page_content[:300]}...\n")