import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Define the FAISS index directory
faiss_index_path = "faiss_index"

# Initialize embedding model
print("Initializing embedding model...")
embedding_model = FastEmbedEmbeddings()

# Check if FAISS index already exists
if os.path.exists(faiss_index_path):
    print("FAISS index found. Loading existing vector store...")
    vector_store = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)
else:
    print("FAISS index not found. Creating new vector store...")

    # Define PDF files
    pdf_files = ["john.pdf", "iphones.pdf", "microsoft.pdf"]

    # Load documents
    print("Loading documents...")
    documents = []
    for pdf in pdf_files:
        loader = PyPDFLoader(f"resources/{pdf}")
        documents.extend(loader.load())

    # Convert documents into text chunks
    print("Initializing text splitter...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    print("Splitting docs content into chunks...")
    split_docs = text_splitter.split_documents(documents)
    texts = [doc.page_content.strip() for doc in split_docs if doc.page_content.strip()]

    # Initialize vector store
    print("Initializing vector store...")
    vector_store = FAISS.from_texts(texts, embedding_model)

    # Store in vector database
    print("Storing document content in vector store...")
    vector_store.save_local(faiss_index_path)

# Initialize retriever
print("Initializing retriever...")
retriever = vector_store.as_retriever(search_kwargs={"k": 1})

# Initialize Gemini llm model
print("Initializing LLM...")
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
llm = GoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

# Create a prompt template
prompt_template = PromptTemplate(
    template="""
    You are an AI assistant. Use the following retrieved documents to answer the user's question.
    If the answer is not found in the documents, say "I don't know" instead of making up an answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """,
    input_variables=["context", "question"],
)

# Define RetrievalQA chain with custom prompt
print("Initialize QA chain...")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template},
)

# Query the RAG system
print("Querying RAG model...")
query = "What is the profession of John?"
response = qa_chain.invoke({"query": query})

# Display the response
print("\n=== Answer ===")
print(response["result"])

print("=== Source Documents ===")
for doc in response["source_documents"]:
    print(f"Document content: {doc.page_content[:300]}...")
    print(f"Document metadata: {doc.metadata}\n")
