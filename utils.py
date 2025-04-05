from datetime import time
import os

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
        print(f"Vector sture index '{pc_index_name}' not found. Create new...")
        create_vector_store(pc, pc_index_name, pdf_files, embedding_model)

    index = pc.Index(pc_index_name)
    return PineconeVectorStore(index=index, embedding=embedding_model)


def create_vector_store(pc, pc_index_name, pdf_files, embedding_model):
    pc.create_index(
        name=pc_index_name,
        dimension=384,  # Replace with your model dimensions
        metric="cosine",  # Replace with your model metric
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
    documents = []
    for pdf in pdf_files:
        loader = PyPDFLoader(f"resources/{pdf}")
        documents.extend(loader.load())
    print(documents)
    split_docs = getTextSplitter().split_documents(documents)
    print(split_docs)
    texts = [doc.page_content.strip() for doc in split_docs if doc.page_content.strip()]
    print(texts)
    vector_store.add_texts(texts)
    return vector_store


def getTextSplitter():
    print("Initializing text splitter...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
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


def getPromptTemplate():
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
    return prompt_template


def createRagChain(llm, vector_store, prompt_template):
    print("\nInitializing QA RAG system...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 1}),
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
        print(f"Document content: {doc.page_content[:300]}...")
        print(f"Document metadata: {doc.metadata}\n")
