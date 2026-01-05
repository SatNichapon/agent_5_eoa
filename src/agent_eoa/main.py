import logging
import sys
from langchain_core.messages import HumanMessage
from .workflow import graph

# Configure Logging to show up in console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def main():
    print("--- Agent 5 (EOA) Initialized ---")
    user_input = "Organize a team meeting on 2023-10-20. If morning is busy, try afternoon."
    
    inputs = {"messages": [HumanMessage(content=user_input)]}
    
    for output in graph.stream(inputs):
        for key, value in output.items():
            print(f"Finished Node: {key}")

if __name__ == "__main__":
    main()