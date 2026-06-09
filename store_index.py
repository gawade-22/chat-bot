import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec 
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_hugging_face_embeddings

# Load environment configurations
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')  # Switched from OPENAI_API_KEY to GROQ_API_KEY

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing from your environment variables setup.")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# 1. Processing Source Medical Documents (Matches your custom helper naming schemas)
extracted_data = load_pdf_files(data='data/')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

# 2. Extract Embeddings Model Layout Vector
# Note: This stays completely local via HuggingFace (all-MiniLM-L6-v2), it does not use Groq or OpenAI!
embeddings = download_hugging_face_embeddings()

# 3. Setting Up Pinecone Client Connectivity
pc = Pinecone(api_key=PINECONE_API_KEY)

# CRITICAL FIX: Must be entirely lowercase to avoid PineconeApiException (400) Bad Request
index_name = "chatbot"  

if not pc.has_index(index_name):
    print(f"Index '{index_name}' not found. Creating a fresh serverless index...")
    pc.create_index(
        name=index_name,
        dimension=384,  # Dimensional structural layout matching sentence-transformers
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    # Give the cloud clusters a brief moment to provision networking infrastructure
    time.sleep(3)  
else:
    print(f"Connected to existing active Pinecone index structure: '{index_name}'")

index = pc.Index(index_name)

# 4. Safe Batched Data Ingestion Pipeline to avoid network latency freezes
print(f"Initializing connection to index. Preparing upload for {len(text_chunks)} text blocks...")
docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# Uploading data in visible segments of 500 records at a time
BATCH_SIZE = 500
for i in range(0, len(text_chunks), BATCH_SIZE):
    batch = text_chunks[i:i + BATCH_SIZE]
    print(f" -> Uploading Batch {i // BATCH_SIZE + 1} (Chunks {i} to {min(i + BATCH_SIZE, len(text_chunks))})...")
    docsearch.add_documents(documents=batch)
    time.sleep(0.5)  # Let local internet socket connections clear safely

print("✅ Success! Your medical vector store index has been successfully populated and compiled.")