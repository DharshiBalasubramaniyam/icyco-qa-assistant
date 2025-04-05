from utils import get_vector_store, getLLM, getPromptTemplate, createRagChain, printResponse
from flask import Flask, request, jsonify

app = Flask(__name__)

vector_store = None
llm = None


@app.post('/chat')
def chat():
    data = request.get_json()

    query = data["query"]
    chat_history = data["chat_history"]

    prompt_template = getPromptTemplate(chat_history)  # Prepare prompt template
    rag_chain = createRagChain(llm, vector_store, prompt_template)  # Initialize rag chain

    response = rag_chain.invoke({"query": query})  # Process the query
    printResponse(response)  # Display the response

    return jsonify({
        "result": response["result"],
    }), 200


if __name__ == '__main__':
    print("Initializing vector store and LLM...")
    pc_index_name = "icyco-ai-assistant"
    pdf_files = ["about.pdf", "products.pdf"]

    vector_store = get_vector_store(pc_index_name, pdf_files)
    llm = getLLM()
    print("Initialization complete!")
    app.run()
