import json
import logging
import os
import sys
from datetime import datetime
from langchain_core.messages import HumanMessage
from agent_eoa.workflow import graph

# --- CONFIGURATION ---
# Dynamically find paths relative to this script
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPUT_FILE = os.path.join(BASE_DIR, "input", "test_cases.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "output", f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Setup File Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_FILE, encoding='utf-8'), # Write to file
        logging.StreamHandler(sys.stdout)                   # Write to console
    ]
)
logger = logging.getLogger("BatchRunner")

def run_batch_tests():
    logger.info(f"🚀 STARTING BATCH TEST RUN")
    logger.info(f"📂 Input: {INPUT_FILE}")
    logger.info(f"📂 Output: {OUTPUT_FILE}")
    logger.info("="*60)

    # 1. Load Test Cases
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        logger.error(f"❌ Input file not found: {INPUT_FILE}")
        return

    # 2. Iterate and Execute
    for case in test_cases:
        case_id = case.get("id", "UNKNOWN")
        prompt = case.get("prompt", "")
        
        logger.info(f"\n🔹 TEST CASE: {case_id} - {case.get('name')}")
        logger.info(f"📝 Prompt: {prompt}")
        logger.info(f"🎯 Expectation: {case.get('expected_behavior')}")
        logger.info("-" * 30)

        inputs = {"messages": [HumanMessage(content=prompt)]}
        step_count = 0
        
        try:
            # Run the Graph
            for output in graph.stream(inputs):
                for node_name, value in output.items():
                    step_count += 1
                    last_msg = value["messages"][-1]
                    
                    if node_name == "planner":
                        content = last_msg.content
                        if last_msg.tool_calls:
                            logger.info(f"   🤖 [PLANNER] Decided to use Tool: {last_msg.tool_calls[0]['name']}")
                        else:
                            logger.info(f"   🤖 [PLANNER] Final Answer: {content}")
                            
                    elif node_name == "tools":
                        # Log tool output
                        logger.info(f"   🛠️  [TOOL OUTPUT] {last_msg.content}")

        except Exception as e:
            logger.error(f"   ❌ CRITICAL ERROR: {e}")

        logger.info(f"✅ Case {case_id} Finished in {step_count} steps.\n")

    logger.info("="*60)
    logger.info("🏁 ALL TESTS COMPLETED")

if __name__ == "__main__":
    run_batch_tests()