from datetime import time
import os
import re

from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()


def get_vector_store(pc_index_name, pdf_files):
    print("Initializing pinecone vector store...")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    embedding_model = getEmbeddingModel()
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if pc_index_name not in existing_indexes:
        print(f"Vector store index '{pc_index_name}' not found. Create new...")
        create_vector_store(pc, pc_index_name, pdf_files, embedding_model)

    index = pc.Index(pc_index_name)
    return PineconeVectorStore(index=index, embedding=embedding_model)


def create_vector_store(pc, pc_index_name, pdf_files, embedding_model):
    pc.create_index(
        name=pc_index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    while not pc.describe_index(pc_index_name).status['ready']:  # Wait for the index to be ready
        time.sleep(1)
    print(f"Successfully created vector store index '{pc_index_name}'. Adding pdf files to new index...")
    index = pc.Index(pc_index_name)
    vector_store = PineconeVectorStore(index=index, embedding=embedding_model)
    add_documents_to_vector_store(vector_store, pdf_files)
    return vector_store


def getTextSplitter():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    return text_splitter


def getEmbeddingModel():
    print("Initializing embedding model...")
    embedding_model = FastEmbedEmbeddings()
    return embedding_model


def getLLM():
    print("Initializing LLM model...")
    api_key = os.getenv("GOOGLE_API_KEY")
    llm = GoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
    return llm


def getPromptTemplate(chat_history):
    print("Initializing prompt template...")
    prompt_template = PromptTemplate(
        template="""
            You are an AI assistant for Icyco, an ice cream shop. Your role is to assist user with questions strictly related to Icyco's products, services, events, or company information.
        
            Use the following retrieved documents to answer the user's question. 
            
            - If the question is unrelated to Icyco, respond with: 
              "I am only able to assist with questions related to Icyco."
            - If the question is related to Icyco but you cannot find the answer in the provided documents, respond with:
              "That’s a great question about Icyco, but I couldn’t find the answer in the available information. You may want to check with Icyco directly for more details."
                    
            Be concise, helpful, and do not make up answers.
        
            Context:
            {context}
            
            """ +
                 f"""
                {chat_history}
                """ +
                 """
                    user: {question}
                    Assistant:
                    """,
        input_variables=["context", "question"]
    )
    return prompt_template


def createRagChain(llm, vector_store, prompt_template):
    print("\nInitializing QA RAG system...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),  # Selects the top 3 most relevant chunks
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template},
    )
    print("The RAG system is successfully initialized. ")
    return qa_chain


def printResponse(response):
    print("\n=== Answer ===")
    print(response["result"])

    print("=== Source Documents ===")
    for doc in response["source_documents"]:
        print(f"\n{doc.metadata['title']}, Page no: {doc.metadata['page_label']}:")
        print(f"Document content: {doc.page_content}...")


def create_chunks(file_path):
    loader = PyPDFLoader(f"resources/{file_path}")
    documents = loader.load()  # Load all pages of the pdf

    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    chunked_docs = getTextSplitter().split_documents(documents)
    return chunked_docs


def add_documents_to_vector_store(vector_store, pdfs):
    chunks = []
    for pdf in pdfs:
        chunks.extend(create_chunks(pdf))
    vector_store.add_documents(chunks)


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
