from utils import get_vector_store, getLLM, getPromptTemplate, createRagChain, printResponse

pc_index_name = "my-first-rag"                            # Define the vector store index
pdf_files = ["john.pdf", "iphones.pdf", "microsoft.pdf"]  # Define pre PDF files

vector_store = get_vector_store(pc_index_name, pdf_files)
llm = getLLM()
prompt_template = getPromptTemplate()

rag_chain = createRagChain(llm, vector_store, prompt_template)

query = input("\nAsk your question: ")
print("Processing...")

response = rag_chain.invoke({"query": query})

# Display the response
printResponse(response)
