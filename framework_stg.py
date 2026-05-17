# corrected_workflow.py
from typing import Annotated, TypedDict, Optional, List, Dict
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from utilities import *
from modules import *

# from langchain_openai import ChatOpenAI  # keep your actual LLM import
# assume helper functions exist:
# NER_RELATION_AGENT, search_chroma, generalize_sparql_with_uri_types,
# build_global_dict, validator, chromadb, Image, display, llm

# ----------------------------
#  STATE STRUCTURE (consistent keys)
# ----------------------------
class State(TypedDict, total=False):
    QUESTION: str
    EXAMPLES: List[str]
    PATTERNS: List[str]
    PATTERNS_LIST: List[str]
    COMPLETED_PROMPTS: Annotated[list, operator.add]
    LLM: Optional[object]  # replace object with ChatOpenAI if available
    LLM2: Optional[object]  # replace object with ChatOpenAI if available
    CONTEXT: Optional[object]
    URIS: Optional[Dict]
    GLOBAL_DICT: Optional[Dict]
    KB: Optional[str]
    FINAL_PROMPT: Optional[str]
    FINAL_SPARQL:Optional[str]

# ----------------------------
#  SPARQL GENERATOR WORKFLOW
#  (define and compile BEFORE using it)
# ----------------------------

def orchestrator(state: State):
    patterns_list = state.get("PATTERNS_LIST") or []
    assert patterns_list is not None, "❌ PATTERNS_LIST cannot be None"
    
    return {"PATTERNS": patterns_list}

def assign_workers(state: State):
    patterns = state.get("PATTERNS") or []
    return [
        Send(
            "llm_call",
            {
                "QUESTION": state.get("QUESTION"),
                "EXAMPLES": state.get("EXAMPLES"),
                "PATTERN": p,
                "LLM": state.get("LLM"),
                "LLM2": state.get("LLM2"),
                "CONTEXT": state.get("CONTEXT"),
                "URIS": state.get("URIS") or {},
                "GLOBAL_DICT": state.get("GLOBAL_DICT") or {},
                "KB": state.get("KB"),
            }
        )
        for p in patterns
    ]

def llm_call(state: State):
    # validator should accept QUESTION, URIS, PATTERN, GLOBAL_DICT, LLM
    query = state.get("PATTERN")
    return {"COMPLETED_PROMPTS": [query]}

def synthesizer(state: State):
    completed = state.get("PATTERN") or []
    return {"COMPLETED_PROMPTS": completed}

# compile SPARQL generator
SPARQL_GEN_STATE = StateGraph(State)
SPARQL_GEN_STATE.add_node("orchestrator", orchestrator)
SPARQL_GEN_STATE.add_node("llm_call", llm_call)
SPARQL_GEN_STATE.add_node("synthesizer", synthesizer)
SPARQL_GEN_STATE.add_edge(START, "orchestrator")
SPARQL_GEN_STATE.add_conditional_edges("orchestrator", assign_workers, ["llm_call"])
SPARQL_GEN_STATE.add_edge("orchestrator", "synthesizer")
SPARQL_GEN_STATE.add_edge("synthesizer", END)
SPARQL_GENERATOR = SPARQL_GEN_STATE.compile()

# ----------------------------
#  MAIN WORKFLOW NODES
# ----------------------------

def NER_Relation_Extraction(state: State):
    print("I am NER")
    """Extract entities and relations as URIs."""
    # NER_RELATION_AGENT should return dict like {"entity": "...", "relation": "..."}
    URIs =  state.get("URIS") #NER_RELATION_AGENT(state.get("KB"), state.get("QUESTION"))
    state["URIS"] = URIs or {}
    return {"URIS": state["URIS"]}

def RAG_AGENT(state: State):
    print("I am RAG")
    """Retrieve relevant knowledge/context and extract patterns."""
    # Example: use chroma client; keep your actual client code
    client = chromadb.PersistentClient(path="./LC_QuAD2.0")
    collection = client.get_collection("qa_collection")
    retrieve = search_chroma(client, state.get("QUESTION"), top_k=10)

    def pattern_sparql(retrieve_list: list):
        patterns = []
        text = ""
        for idx, item in enumerate(retrieve_list):
            sparql_q = item.get('sparql_query') if isinstance(item, dict) else None
            #gen = replace_entity_relation_uris_function_complete(sparql_q) if sparql_q else ""#QALD9
            #gen = replace_uris_wiki(sparql_q) if sparql_q else ""#QALD9
            gen = replace_sparql_uris_QALD9(sparql_q) if sparql_q else ""#QALD9
            #gen = replace_uris_with_placeholder(sparql_q) if sparql_q else ""#VQUANDA
            text += f'--> Pattern {idx} : "{gen}"\n'
            patterns.append(gen)
        return text, patterns

    def extract_patterns(pattern_text: str) -> list:
        patterns = []
        for line in pattern_text.splitlines():
            line = line.strip()
            if line.startswith("--> Pattern"):
                _, pattern = line.split(":", 1)
                print(pattern)
                patterns.append(pattern.strip().strip('"'))
        return patterns

    examples = CONTEXT_CREATOR_DYNAMIC(retrieve)

    pattern_text,patterns = pattern_sparql(retrieve or [])
    state["EXAMPLES"] = examples
    state["PATTERNS"] = pattern_text
    state["CONTEXT"] = retrieve
    state["GLOBAL_DICT"] = build_global_dict(retrieve or [])
    state["PATTERNS_LIST"] =  patterns#extract_patterns(pattern_text)
    return {
        "EXAMPLES": state["EXAMPLES"],
        "PATTERNS": state["PATTERNS"],
        "CONTEXT": state["CONTEXT"],
        "PATTERNS_LIST": state["PATTERNS_LIST"],
        "GLOBAL_DICT": state["GLOBAL_DICT"]
    }

def GENERATION_AGENT(state: State):
    print("I am Generation")
    """Generate SPARQL query from question + URIs + context using SPARQL_GENERATOR."""
    # Defensive reads
    patterns_list = state.get("PATTERNS_LIST") or []
    uris = state.get("URIS") or {}
    global_dict = state.get("GLOBAL_DICT") or {}
    llm_1 = state.get("LLM")
    llm_2 = state.get("LLM2")

    # build a test_state compatible with SPARQL_GENERATOR
    test_state = {
        "QUESTION": state.get("QUESTION"),
        "EXAMPLES": state.get("EXAMPLES"),
        "PATTERNS_LIST": patterns_list,
        "COMPLETED_PROMPTS": [],
        "URIS": uris,
        "GLOBAL_DICT": global_dict,
        "LLM": llm_1,
        "LLM2": llm_2,
        "KB": state.get("KB"),
        "CONTEXT": state.get("CONTEXT") or ""
    }

    # invoke the SPARQL generator workflow
    result = SPARQL_GENERATOR.invoke(test_state)

    # for simple use, take the first generated query (if any)
    completed = result.get("COMPLETED_PROMPTS") or []
    first_query = ""#validator2(state['QUESTION'],state['EXAMPLES'],state['URIS'],state['PATTERNS_LIST'],state['GLOBAL_DICT'],state['LLM'])
     
    state["COMPLETED_PROMPTS"] = completed

    return {"COMPLETED_PROMPTS": completed}

import time
def JUDGE_AGENT(state: State):
    print("I am Judge")
    start = time.time()
    """Generate SPARQL query from question + URIs + context using SPARQL_GENERATOR."""
    final = GENERATOR_AGENT(nl_query = state['QUESTION'],
                            examples = state["EXAMPLES"],
                            SPARQL_LIST = state['COMPLETED_PROMPTS'],
                            URIS = state['URIS'],
                            Global_Dict = state['GLOBAL_DICT'],
                            context = state["CONTEXT"],
                            llm = state["LLM"],
                            llm2 = state["LLM2"],
                           )
    state["FINAL_PROMPT"] = final
    end = time.time()

    print(end-start)

    return {"FINAL_PROMPT": final}



def SPARQL_AGENT(state: State):
    print("I am SPARQL")
    """Generate SPARQL query from question + URIs + context using SPARQL_GENERATOR."""
    final_sparql = GENERATOR_AGENT3(nl_query = state['QUESTION'],
                            examples = state["EXAMPLES"],
                            URIs = state['URIS'],
                            Pattern = state['FINAL_PROMPT'],
                            Global_Dict = state['GLOBAL_DICT'],
                            llm = state["LLM"],
                            llm2 = state["LLM2"],
                           )

    return {"FINAL_SPARQL": final_sparql}
# ----------------------------
#  BUILD MAIN WORKFLOW
# ----------------------------

workflow = StateGraph(State)
workflow.add_node("NER_Relation_Extraction", NER_Relation_Extraction)
workflow.add_node("RAG_AGENT", RAG_AGENT)
workflow.add_node("GENERATION_AGENT", GENERATION_AGENT)
workflow.add_node("JUDGE_AGENT", JUDGE_AGENT)
workflow.add_node("SPARQL_AGENT", SPARQL_AGENT)
workflow.add_edge(START, "NER_Relation_Extraction")
workflow.add_edge("NER_Relation_Extraction", "RAG_AGENT")
workflow.add_edge("RAG_AGENT", "GENERATION_AGENT")
workflow.add_edge("GENERATION_AGENT", "JUDGE_AGENT")
workflow.add_edge("JUDGE_AGENT", "SPARQL_AGENT")
workflow.add_edge("SPARQL_AGENT", END)
CHAIN = workflow.compile()

# ----------------------------
#  RUN THE WORKFLOW (example)
if __name__ == "__main__":
    pass