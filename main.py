from langchain.agents import initialize_agent, AgentType
from tools import tools
from utils import getLLM

llm = getLLM()
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

response = agent.run("What is 12 * 8?")
print("\nFinal Answer:", response)

response = agent.run("Reverse word 'Apple'.")
print("\nFinal Answer:", response)
