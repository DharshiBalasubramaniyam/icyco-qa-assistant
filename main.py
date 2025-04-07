from utils import get_vector_store, getLLM, getPromptTemplate, createRagChain, printResponse

pc_index_name = "icyco-qa-assistant"                            # Define the vector store index
pdf_files = ["about.pdf", "products.pdf", "faqs.pdf"]           # Define pre PDF files

vector_store = get_vector_store(pc_index_name, pdf_files)       # Initialize vector store
llm = getLLM()                                                  # Initialize llm
prompt_template = getPromptTemplate()                           # Prepare prompt template

rag_chain = createRagChain(llm, vector_store, prompt_template)  # Initialize rag chain

while True:
    query = input("\nYou: ")                                    # Input query from user
    if query.strip().lower() == "exit":                         # Exit from chat
        print("Assistant: Bye")
        break

    response = rag_chain.invoke({"query": query})               # Process the query
    print(f"Assistant: {response['result']}")                   # Display the response
