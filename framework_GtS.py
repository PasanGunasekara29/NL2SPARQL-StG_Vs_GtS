from typing import Annotated, TypedDict
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_openai import ChatOpenAI
from utilities import *
from modules_GtS import *

# =========================================================
#  STATE STRUCTURES
# =========================================================

class State(TypedDict, total=False):
    QUESTION: str
    EXAMPLES: list[str]
    PATTERNS: list[str]
    PATTERNS_LIST: list
    COMPLETED_SPARQLS: Annotated[list, operator.add]
    LLM: ChatOpenAI | None
    CONTEXT: str | None
    URIs: dict | None
    GLOBAL_DICT: dict | None
    KB: str | None


class WorkerState(TypedDict, total=False):
    QUESTION: str
    EXAMPLES: list[str]
    PATTERN: str
    COMPLETED_SPARQLS: Annotated[list, operator.add]
    LLM: ChatOpenAI | None
    CONTEXT: str | None
    URIs: dict | None
    GLOBAL_DICT: dict | None
    KB: str | None


def orchestrator(state: State):

    assert state["PATTERNS_LIST"] is not None, "❌ PATTERNS_LIST cannot be None"

    print(f"\n🟦 ORCHESTRATOR RECEIVED {len(state['PATTERNS_LIST'])} PATTERNS")

    return {
        "PATTERNS": state["PATTERNS_LIST"]
    }


def assign_workers(state: State):

    patterns = state["PATTERNS"]

    print(f"\n🟨 ASSIGNING {len(patterns)} WORKERS...")

    return [
        Send(
            "llm_call",
            {
                "QUESTION": state["QUESTION"],
                "PATTERN": p,
                "LLM": state.get("LLM"),
                "CONTEXT": state.get("CONTEXT"),
                "URIs": state.get("URIs") or {},
                "GLOBAL_DICT": state.get("GLOBAL_DICT") or {},
                "KB": state.get("KB"),
            }
        )
        for p in patterns
    ]


def llm_call(state: WorkerState):

    query = validator(
        state["QUESTION"],
        state["EXAMPLES"],
        state.get("URIs") or {},
        state["PATTERN"],
        state.get("GLOBAL_DICT") or {},
        state.get("LLM")
    )

    print(f"   🔹 GENERATED: {query}")

    return {"COMPLETED_SPARQLS": [query]}


def synthesizer(state: State):

    print(f"\n🟩 SYNTHESIZER MERGED {len(state['COMPLETED_SPARQLS'])} QUERIES")

    return {
        "COMPLETED_SPARQLS": state["COMPLETED_SPARQLS"]
    }

workflow = StateGraph(State)

workflow.add_node("orchestrator", orchestrator)
workflow.add_node("llm_call", llm_call)
workflow.add_node("synthesizer", synthesizer)
workflow.add_edge(START, "orchestrator")
workflow.add_conditional_edges(
    "orchestrator",
    assign_workers,
    ["llm_call"]
)

workflow.add_edge("orchestrator", "synthesizer")
workflow.add_edge("synthesizer", END)
SPARQL_GENERATOR = workflow.compile()


# corrected_workflow.py
from typing import Annotated, TypedDict, Optional, List, Dict
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
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
    COMPLETED_SPARQLS: Annotated[list, operator.add]
    LLM: Optional[object]  # replace object with ChatOpenAI if available
    CONTEXT: Optional[object]
    URIS: Optional[Dict]
    GLOBAL_DICT: Optional[Dict]
    KB: Optional[str]
    SPARQL: Optional[str]
    FINAL_SPARQL: Optional[str]

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
    query = validator(
        state.get("QUESTION"),
        state.get("EXAMPLES"),
        state.get("URIS") or {},
        state.get("PATTERN"),
        state.get("GLOBAL_DICT") or {},
        state.get("LLM")
    )
    return {"COMPLETED_SPARQLS": [query]}

def synthesizer(state: State):
    completed = state.get("COMPLETED_SPARQLS") or []
    return {"COMPLETED_SPARQLS": completed}

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
    """Extract entities and relations as URIs."""
    # NER_RELATION_AGENT should return dict like {"entity": "...", "relation": "..."}
    #URIs = NER_RELATION_AGENT(state.get("KB"), state.get("QUESTION"))
    state["URIS"] = state.get("URIS")#URIs or {}
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
            #gen = replace_entity_relation_uris_custom(sparql_q) if sparql_q else ""#QALD9
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
    """Generate SPARQL query from question + URIs + context using SPARQL_GENERATOR."""
    # Defensive reads
    patterns_list = state.get("PATTERNS_LIST") or []
    uris = state.get("URIS") or {}
    global_dict = state.get("GLOBAL_DICT") or {}
    llm_ref = state.get("LLM")

    # build a test_state compatible with SPARQL_GENERATOR
    test_state = {
        "QUESTION": state.get("QUESTION"),
        "EXAMPLES": state.get("EXAMPLES"),
        "PATTERNS_LIST": patterns_list,
        "COMPLETED_SPARQLS": [],
        "URIS": uris,
        "GLOBAL_DICT": global_dict,
        "LLM": llm_ref,
        "KB": state.get("KB"),
        "CONTEXT": state.get("CONTEXT") or ""
    }

    # invoke the SPARQL generator workflow
    result = SPARQL_GENERATOR.invoke(test_state)

    # for simple use, take the first generated query (if any)
    completed = result.get("COMPLETED_SPARQLS") or []
    first_query = ""#validator2(state['QUESTION'],state['EXAMPLES'],state['URIS'],state['PATTERNS_LIST'],state['GLOBAL_DICT'],state['LLM'])

    state["SPARQL"] = first_query
    state["COMPLETED_SPARQLS"] = completed

    return {"SPARQL": first_query, "COMPLETED_SPARQLS": completed}

def JUDGE_AGENT(state: State):
    """Generate SPARQL query from question + URIs + context using SPARQL_GENERATOR."""
    final = GENERATOR_AGENT(state['QUESTION'],state["EXAMPLES"],state['COMPLETED_SPARQLS'],state['SPARQL'],state['URIS'],state['GLOBAL_DICT'],state["CONTEXT"],state["LLM"])

    state["FINAL_SPARQL"] = final

    return {"FINAL_SPARQL": final}

# ----------------------------
#  BUILD MAIN WORKFLOW
# ----------------------------

workflow = StateGraph(State)
workflow.add_node("NER_Relation_Extraction", NER_Relation_Extraction)
workflow.add_node("RAG_AGENT", RAG_AGENT)
workflow.add_node("GENERATION_AGENT", GENERATION_AGENT)
workflow.add_node("JUDGE_AGENT", JUDGE_AGENT)
workflow.add_edge(START, "NER_Relation_Extraction")
workflow.add_edge("NER_Relation_Extraction", "RAG_AGENT")
workflow.add_edge("RAG_AGENT", "GENERATION_AGENT")
workflow.add_edge("GENERATION_AGENT", "JUDGE_AGENT")
workflow.add_edge("JUDGE_AGENT", END)
CHAIN = workflow.compile()



if __name__ == "__main__":
   pass
    

    