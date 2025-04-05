from utils import get_vector_store, getLLM, getPromptTemplate, createRagChain, printResponse

pc_index_name = "my-first-rag"                                  # Define the vector store index
pdf_files = ["about.pdf", "products.pdf"]                       # Define pre PDF files

vector_store = get_vector_store(pc_index_name, pdf_files)       # Initialize vector store
llm = getLLM()                                                  # Initialize llm
prompt_template = getPromptTemplate()                           # Prepare prompt template

rag_chain = createRagChain(llm, vector_store, prompt_template)  # Initialize rag chain

query = input("\nAsk your question: ")                          # Input query from user, e.g. Do icyco take catering orders?
print("Processing...")

response = rag_chain.invoke({"query": query})                   # Process the query
printResponse(response)                                         # Display the response
