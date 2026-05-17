from utilities import *
from modules import *
from framework_stg import *
from evaluation import *
import spacy_component

if __name__ == "__main__":
    #df_test = pd.read_csv("output_10.csv")
    ##vector_db("train.json","question","query") # Use to create vector database
    
    llm = ChatOpenAI(
    model_name="hf.co/bartowski/Qwen2.5-14B-Instruct-GGUF:Qwen2.5-14B-Instruct-Q8_0.gguf",
    base_url="http://localhost:11434/v1",
    openai_api_key="fhguh",
)
    llm2 = ChatOpenAI(
    model_name="hf.co/bartowski/Qwen2.5-14B-Instruct-GGUF:Qwen2.5-14B-Instruct-Q8_0.gguf",
    base_url="http://localhost:11434/v1",
    openai_api_key="fhguh",
)
    
    initial_state = {
        "QUESTION": "Who won both Nobel Prize and Oscars?",#df_test["question"][0],
        "KB": "dbpedia",
        "LLM": llm,  # set your LLM object here
        "LLM2" :llm2
    }


