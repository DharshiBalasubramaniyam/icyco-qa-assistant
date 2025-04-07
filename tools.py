from langchain.agents import initialize_agent, Tool


def calculator_tool(query: str) -> str:
    print("\nUsing Calculator tool...")
    try:
        result = eval(query)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# Tool 2: Text Reverser
def reverse_text_tool(text: str) -> str:
    print("\nUsing TextReverser tool...")
    return text[::-1]


tools = [
    Tool(
        name="Calculator",
        func=calculator_tool,
        description="Useful for doing math calculations. Input should be a valid Python expression like '2 + 2'."
    ),
    Tool(
        name="TextReverser",
        func=reverse_text_tool,
        description="Useful for reversing a string. Input should be a sentence or word."
    )
]
