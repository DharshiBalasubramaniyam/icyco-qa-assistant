import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

load_dotenv()

# Define the FAISS index directory
faiss_index_path = "faiss_index"

# Define pre PDF files
pdf_files = ["john.pdf", "iphones.pdf", "microsoft.pdf"]

print("Initializing text splitter...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

print("Initializing embedding model...")
embedding_model = FastEmbedEmbeddings()

print("Initializing vector store...")
if os.path.exists(faiss_index_path):
    print("FAISS index found. Loading existing vector store...")
    vector_store = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)
else:
    print("FAISS index not found. Creating new vector store with pre pdf files...")

    documents = []
    for pdf in pdf_files:
        loader = PyPDFLoader(f"resources/{pdf}")
        documents.extend(loader.load())

    split_docs = text_splitter.split_documents(documents)
    texts = [doc.page_content.strip() for doc in split_docs if doc.page_content.strip()]

    vector_store = FAISS.from_texts(texts, embedding_model)

    print("Storing new vector store...")
    vector_store.save_local(faiss_index_path)

print("Initializing LLM model...")
api_key = os.getenv("GOOGLE_API_KEY")
llm = GoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

print("Initializing prompt template...")
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

print("Initializing QA chain...")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 1}),
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template},
)
print("The QA chain is successfully initialized. ")

query = input("\nAsk your question: ")
print("Processing...")
response = qa_chain.invoke({"query": query})

# Display the response
print("\n=== Answer ===")
print(response["result"])

print("=== Source Documents ===")
for doc in response["source_documents"]:
    print(f"Document content: {doc.page_content[:300]}...")
    print(f"Document metadata: {doc.metadata}\n")

