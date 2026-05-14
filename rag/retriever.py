import os
import json
import glob
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.documents import Document
from supabase.client import create_client, Client

# Load Environment Variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def get_supabase_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise ValueError("Missing Supabase credentials in environment variables.")
    try:
        return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    except Exception as e:
        raise ConnectionError(f"Failed to connect to Supabase: {e}")

def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    if not GEMINI_API_KEY:
        raise ValueError("Missing GEMINI_API_KEY in environment variables.")
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)

def ingest_knowledge_base(kb_dir: str = "data/kb/"):
    """
    Reads JSON files, splits text, and stores in Supabase vector store.
    """
    if not os.path.exists(kb_dir):
        print(f"Knowledge base directory '{kb_dir}' not found.")
        return
        
    documents = []
    json_files = glob.glob(os.path.join(kb_dir, "*.json"))
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            doc_id = data.get("id")
            title = data.get("title")
            content = data.get("content")
            
            if not doc_id or not title or not content:
                print(f"Skipping {file_path}: Missing id, title, or content.")
                continue
                
            # Create a Document
            doc = Document(
                page_content=content,
                metadata={"id": doc_id, "title": title}
            )
            documents.append(doc)
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    if not documents:
        print("No valid documents found to ingest.")
        return

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)
    
    try:
        supabase_client = get_supabase_client()
        embeddings = get_embeddings()
        
        # Note: the table_name depends on user setup. Usually it's 'documents' by default.
        SupabaseVectorStore.from_documents(
            split_docs,
            embeddings,
            client=supabase_client,
            table_name="documents",
            query_name="match_documents"
        )
        print(f"Successfully ingested {len(split_docs)} document chunks into Supabase.")
    except Exception as e:
        print(f"Failed to ingest knowledge base: {e}")

def search_kb(query: str) -> str:
    """
    Performs a similarity search on the Supabase vector store and formats the retrieved documents.
    """
    try:
        supabase_client = get_supabase_client()
        embeddings = get_embeddings()
        
        vector_store = SupabaseVectorStore(
            client=supabase_client,
            embedding=embeddings,
            table_name="documents",
            query_name="match_documents"
        )
        
        # Perform similarity search
        retrieved_docs = vector_store.similarity_search(query, k=5)
        
        if not retrieved_docs:
            return "No relevant documents found."
            
        # Format the retrieved documents
        formatted_docs = []
        for doc in retrieved_docs:
            source_id = doc.metadata.get("id", "Unknown")
            title = doc.metadata.get("title", "Untitled")
            content = doc.page_content
            
            # Strict Rule: [Source: {metadata['id']}] {title}: {content}
            formatted_string = f"[Source: {source_id}] {title}: {content}"
            formatted_docs.append(formatted_string)
            
        return "\n\n".join(formatted_docs)
        
    except Exception as e:
        return f"Error during retrieval: {e}"
