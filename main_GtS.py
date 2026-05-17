from utilities import *
from modules_GtS import *
from framework_GtS import *
from evaluation import *
import spacy_component

if __name__ == "__main__":
    llm = ChatOpenAI(
    model_name="hf.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF:qwen2.5-1.5b-instruct-q8_0.gguf",
    base_url="http://localhost:11434/v1",
    openai_api_key="fhguh",
)
    
    initial_state = {
        "QUESTION": "Who won both Oscars and Nobel Prize?",
        "KB": "dbpedia",
        "LLM": llm,  # set your LLM object here
    }
    final_state = CHAIN.invoke(initial_state)
    for q in final_state.get("COMPLETED_SPARQLS", []):
        print(q)