import os
from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq  # Switched from langchain_openai to langchain_groq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import system_prompt  

app = Flask(__name__)

# Load environment configurations from your .env file
from dotenv import load_dotenv
load_dotenv()

# Extract keys cleanly 
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Download the specific embeddings tested in the notebook environment
embeddings = download_hugging_face_embeddings()

# Name of the pinecone index populated during the trials stage
index_name = "chatbot" 

# Connect to your existing loaded vector database index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Configuration matching the similarity retrieval setup
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize the Free Groq model configuration using Meta Llama 3.1
chatModel = ChatGroq(model="llama-3.1-8b-instant", temperature=0.4)

# Prompt setup containing the specialized Medical Assistant system persona
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Constructing the RAG execution chain components
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    # Utilizing request.form.get ensures type safety and prevents empty field crashes
    msg = request.form.get("msg", "")
    if not msg:
        return "Please enter a valid message.", 400
        
    print(f"User Input: {msg}")
    
    # Invoking the Retrieval-Augmented Generation workflow pipeline
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    
    return str(response["answer"])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)