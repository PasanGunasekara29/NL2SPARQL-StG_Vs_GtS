from utilities import *
import spacy_component
from utilities import *
# ==============================
# Standard Libraries
# ==============================
import time
import operator
import requests
from typing import Annotated, List
from typing_extensions import TypedDict, Literal

# ==============================
# Data Processing
# ==============================
import pandas as pd
import json
# ==============================
# NLP & Embeddings
# ==============================
import spacy
import spacy_dbpedia_spotlight
from sentence_transformers import SentenceTransformer
import spacy_component

# ==============================
# Vector Database
# ==============================
import chromadb

# ==============================
# LangChain Core
# ==============================
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import SystemMessage, HumanMessage

# ==============================
# LangChain Community
# ==============================
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ==============================
# LangChain OpenAI
# ==============================
from langchain_openai import ChatOpenAI

# ==============================
# LangGraph
# ==============================
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

# ==============================
# Pydantic
# ==============================
from pydantic import BaseModel, Field

# ==============================
# Jupyter Display
# ==============================
from IPython.display import Image, display
from utilities import *

# ==============================
# Project Modules
# ==============================
from Tools.rag_tools import search_chroma
from RAG_Agent import run_rag_agent
from NER_Agent import entity_linking, relation_linking

def NER_RELATION_AGENT(kb:str, question:str):
  uris = wikidata_labels_from_pipeline(URI_Extraction(kb, question)) #URI_Extraction(kb, question) DBpedia KG
  return uris

def GENERATOR_AGENT2(nl_query: str,example:list, URIs: dict, Patterns: str, Global_Dict: dict, llm: ChatOpenAI) -> str:
    if URIs:
      entity = URIs.get("entity")
      relation = URIs.get("relation")
    else:
      URIs = {}
    if Global_Dict:
      external_uri = Global_Dict.get("entity")
      external_relation = Global_Dict.get("relation")
      tripple = Global_Dict.get("triples")
    else:
      Global_Dict = {}
    text = ""
    for items in example:
      text += items
      text += "\n"
      text += "\n"

    prompt2 = """You are an assistant that constructs SPARQL queries based on a given natural language question, predefined query patterns, and provided URIs.

Follow these strict rules:

- PATTERNS and URIs MUST be selected based on the details given with respect to the question.
- FIRST, pick the SPARQL pattern that best matches the question's meaning using the provided Entity URIs and Relation URIs.
- EMBED the given URI TAGs logically into the placeholders within the chosen pattern.
- USE ONLY the given URI TAGs. Do NOT invent, assume, or modify any URI TAGs.
- DO NOT alter the structure or syntax of the chosen SPARQL pattern, except for replacing the URI placeholders.
- ENSURE grammatical and logical correctness in how the URI TAGs are inserted.
- OUTPUT only the final SPARQL query — no reasoning, no explanations, and no commentary.

<example>
EXAMPLE SCENARIO:
{examples}
</example>

USE THE ABOVE THINKING PATTERN TO GENERATE A SPARQL QUERY FOR THE FOLLOWING QUESTION:

[QUESTION]:
{Question}

PATTERNS AND URIs MUST BE SELECTED BASED ON THE DETAILS GIVEN WITH RESPECT TO THE QUESTION.

INPUTS:

[PATTERNS]:
{Patterns}

[ENTITY URIs]:
{Entity_URIs}

[RELATION URIs]:
{Relation_URIs}

ADDITIONAL DATA FOR FINAL CORRECTION:

EXTERNAL TRIPLES:
{tripple}

OUTPUT FORMAT:

- Return ONLY the final SPARQL query with URI TAGs (NOT URLs such as http links).
- Return ONLY valid JSON with the field "sparql".
- No text outside JSON.
- No explanations.

{{SPARQL}}
"""

    PROMPT = PromptTemplate(
        template=prompt2,
        input_variables=["examples","Question", "Patterns","Entity_URIs", "Relation_URIs", "tripple"]
    )

    formatted_prompt = PROMPT.format(
        examples=text,
        Patterns=Patterns,
        Entity_URIs=entity,
        Relation_URIs=relation,
        Question=nl_query,
        tripple = tripple
    )
    messages = [
        SystemMessage(content="You an expert to convert natural language questions to SPARQL queries through reasoning, but output only the final query."),
        HumanMessage(content=formatted_prompt)
    ]
    llm = llm.with_structured_output(SPARQLQuery)

    response = llm.invoke(messages)
    return response.sparql


def GENERATOR_AGENT(nl_query: str,examples:list,SPARQL_LIST : list, SPARQL: str, URIS:dict, Global_Dict:dict, context:str, llm: ChatOpenAI, max_iterations:int=1) -> str:

    if URIS:
        entity = URIS.get("entity")
        relation = URIS.get("relation")
    else:
        entity = None
        relation = None

    text = ""
    for items in examples:
        text += items
        text += "\n"
        text += "\n"

    sparql_list = SPARQL_CONTEXT(SPARQL_LIST,SPARQL)

    prompt5 = """You are an expert at selecting the correct SPARQL query from a provided list of SPARQL query candidates based on a natural language question and given URIs. Follow all rules exactly.

TASK:
Select ONLY THE CORRECT SPARQL query from the provided list.

You MUST reason internally and pick the SPARQL query that is most aligned with the natural language question from the given list. Do NOT generate new queries.

You are ONLY allowed to pick ONE SPARQL query from the provided list of SPARQL queries that best matches the given details.

PICK the BEST SPARQL query that matches the given details from the "List of SPARQL queries given".

<Think and Follow>

STRICT RULES:
1. The chosen SPARQL query must be exactly one of the SPARQL queries from the provided list.
2. Do NOT modify, rewrite, or edit the selected SPARQL query.
3. Do NOT generate new URIs.
4. Do NOT output explanations or reasoning steps.
5. Output ONLY the final selected SPARQL query.

SELECTION GUIDELINES:
1. Match the meaning of the question with the generated list of SPARQL queries.
2. Match the entity type (country, cave, place, language, mountain, etc.).
3. Match numeric constraints (e.g., “more than two”, “more than 10”).
4. Match relation types (location, places, official language, etc.).

Use ONLY the given URIs:
- The selected query must be compatible with the Entity URI tags, Relation URI tags, and External URIs.
- Never introduce a URI not found in the given context.

Match the Examples Context:
- Use correct ontology classes (dbo:Cave, dbo:Country, etc.).
- Use correct relations (dbo:location, dbo:country, etc.).
- Ensure counting patterns follow GROUP BY and HAVING if required.

OUTPUT FORMAT:
- Output ONLY the chosen SPARQL query.
- No explanations, commentary, or reasoning.

FORBIDDEN:
- Inventing URIs
- Editing or rewriting SPARQL queries
- Combining multiple SPARQL queries
- Returning more than one query
- Generating new URIs
- Generating new triple patterns
- Adding comments or explanations

FINAL OUTPUT:
Return exactly ONE SPARQL query from the list of SPARQL queries that best aligns with the given question.

Context:
{context}

Examples:
{examples}

Entity URIs:
{entity_uris}

Relation URIs:
{relation_uris}

External URIs:
{external_uris}

List of SPARQL queries given:
{SPARQL_LIST}

Given Question:
{question}

Output ONLY the final SPARQL query.

Return ONLY valid JSON with the field "sparql".

Rules for output:
- No URLs such as http links in the output.
- No explanations.
- No text outside JSON.
- No additional fields.


</Think and Follow>

OUTPUT SPARQL QUERY PICKED FROM List of SPARQL queries given:
{{SPARQL QUERY PICKED FROM THE GIVEN LIST}}
"""

    context_text = context

    PROMPT = PromptTemplate(
        template=prompt5,
        input_variables=["context","examples","entity_uris","relation_uris","external_uris","SPARQL_LIST","question"]
    )

    formatted_prompt = PROMPT.format(
        context=context_text,
        examples=text,
        entity_uris=entity,
        relation_uris=relation,
        external_uris=Global_Dict,
        SPARQL_LIST=sparql_list,
        question=nl_query
    )

    messages = [
        SystemMessage(content="You an expert picking correct sparql query from given list of SPARQL queries that correctly align with given natural laguage question."),
        HumanMessage(content=formatted_prompt)
    ]

    llm_select = llm.with_structured_output(SPARQLQuerySelect)

    generated = llm_select.invoke(messages)

    current_query = generated.sparql

    # ---------- Evaluator Prompt (Minimal change) ----------

    evaluator_prompt = prompt5 + """

CURRENT QUERY SELECTED:
{current_query}

Verify if this query is the correct one from the list.
If it is correct return it unchanged.
If it is incorrect select the correct query from the given list.
"""

    evaluator_template = PromptTemplate(
        template=evaluator_prompt,
        input_variables=["context","examples","entity_uris","relation_uris","external_uris","SPARQL_LIST","question","current_query"]
    )

    for _ in range(max_iterations):

        formatted_eval = evaluator_template.format(
            context=context_text,
            examples=text,
            entity_uris=entity,
            relation_uris=relation,
            external_uris=Global_Dict,
            SPARQL_LIST=sparql_list,
            question=nl_query,
            current_query=current_query
        )

        eval_messages = [
            SystemMessage(content="Evaluate the selected SPARQL query."),
            HumanMessage(content=formatted_eval)
        ]

        evaluated = llm_select.invoke(eval_messages)

        if evaluated.sparql.strip() == current_query.strip():
            break

        current_query = evaluated.sparql

    response = SPARQLQuerySelect(sparql=current_query)

    return response.sparql

def GENERATOR_AGENT3(
    nl_query: str,
    examples: list,
    URIs: dict,
    Pattern: str,
    Global_Dict: dict,
    llm: ChatOpenAI,
    max_iterations: int = 5
) -> str:

    start = time.time()

    if URIs:
        entity = URIs.get("entity")
        relation = URIs.get("relation")
    else:
        entity = None
        relation = None

    if Global_Dict:
        external_uri = Global_Dict.get("entity")
        external_relation = Global_Dict.get("relation")
        tripple = Global_Dict.get("triples")
    else:
        tripple = None

    text = ""
    for items in examples:
        text += items
        text += "\n\n"

    # ---------------- GENERATOR PROMPT ---------------- #

    generator_prompt = """You are an expert that constructs SPARQL queries based on a given natural language question. 
You have given predefined SPARQL query patterns, and Entity and Relation URIs to generate SPARQL queries.
Follow these strict rules:

- PATTERNS AND URIs MUST BE PICKED based on the details given with respect to the question.
- USE the given pattern to generate SPARQL queries by embedding Entity URIs and Relation URIs.
- EMBED the given URIs logically into the placeholders within the given pattern.

URI REPLACEMENT RULES:
- Replace {{Replace With URIs}} with the provided Entity URIs.
- Replace {{Replace With URIs}} with the provided Relation URIs.
- Ensure the URIs are placed correctly in subject, predicate, or object positions according to their type.
- Replace Relation or Entity URIs with better External URIs if explicitly provided in the External Triples section.

STRICT CONSTRAINTS:
- USE ONLY the given URIs.
- Do NOT invent, assume, or modify any URI.
- Do NOT alter the structure or syntax of the chosen SPARQL pattern, except for replacing the URI placeholders.
- Ensure grammatical and logical correctness when inserting URIs.
- ALL URI placeholders in the pattern MUST be filled using the provided URIs.

OUTPUT REQUIREMENTS:
- Output ONLY the final SPARQL query.
- No explanations.
- No commentary.

HARD RULES:
- USING THE GIVEN PATTERN IS COMPULSORY.
- PATTERN CAN MERELY CHANGE BASED ON REASONING ABOUT NATURAL LANGUAGE QUESTION.
- USE ONLY GIVEN ENTITY AND RELATION URIs.
- DO NOT invent, assume, or modify any URIs.

EXAMPLE SCENARIOS:
{examples}

USE THE ABOVE THINKING PATTERN TO GENERATE A SPARQL QUERY FOR THE FOLLOWING QUESTION:

Question:
{Question}

PATTERNS AND URIs MUST BE SELECTED BASED ON THE DETAILS GIVEN WITH RESPECT TO THE QUESTION.

INPUTS:

SHOULD USE THIS PATTERN:
{Pattern}

ENTITY URIs:
{Entity_URIs}

RELATION URIs:
{Relation_URIs}

ADDITIONAL DATA FOR FINAL CORRECTION:

EXTERNAL TRIPLES:
{tripple}

OUTPUT FORMAT:
- Return ONLY valid JSON with the field "sparql".
- No URLs such as http links in the output.
- No text outside JSON.
- No explanations.
- The final SPARQL query must contain URI TAGs, not full URLs such as https.


{{SPARQL QUERY}}
"""

    generator_template = PromptTemplate(
        template=generator_prompt,
        input_variables=[
            "examples",
            "Question",
            "Pattern",
            "Entity_URIs",
            "Relation_URIs",
            "tripple"
        ]
    )

    formatted_generator = generator_template.format(
        examples=text,
        Question=nl_query,
        Pattern=Pattern,
        Entity_URIs=entity,
        Relation_URIs=relation,
        tripple=tripple
    )

    generator_messages = [
        SystemMessage(content="Generate a SPARQL query using the given pattern."),
        HumanMessage(content=formatted_generator)
    ]

    generator_llm = llm.with_structured_output(SPARQLQuery)

    generated = generator_llm.invoke(generator_messages)

    current_query = generated.sparql

    # ---------------- EVALUATOR PROMPT ---------------- #

    evaluator_prompt = """You are an assistant that evaluates SPARQL queries based on a given natural language question, predefined query patterns, and provided URIs.

CURRENT QUERY:
{current_query}

PREVIOUS ATTEMPTS HISTORY:
{attempt_history}

CURRENT ITERATION: {iteration} of {max_iterations}

Use the attempt history to avoid repeating the same mistakes. If you have seen this query before or a similar one that did not satisfy the pattern, try a different approach.

Follow these strict rules:

- PATTERNS AND URIs MUST BE PICKED based on the details given with respect to the question.
- USE the given pattern to generate SPARQL queries by embedding Entity URIs and Relation URIs.
- EMBED the given URIs logically into the placeholders within the given pattern.

URI REPLACEMENT RULES:
- Replace {{Replace With Entity URIs}} with the provided Entity URIs.
- Replace {{Replace With Relation URIs}} with the provided Relation URIs.
- Ensure the URIs are placed correctly in subject, predicate, or object positions according to their type.
- Replace Relation or Entity URIs with better External URIs if explicitly provided in the External Triples section.

STRICT CONSTRAINTS:
- USE ONLY the given URIs.
- Do NOT invent, assume, or modify any URI.
- Do NOT alter the structure or syntax of the chosen SPARQL pattern, except for replacing the URI placeholders.
- Ensure grammatical and logical correctness when inserting URIs.
- ALL URI placeholders in the pattern MUST be filled using the provided URIs.

OUTPUT REQUIREMENTS:
- Output ONLY the final SPARQL query.
- No explanations.
- No commentary.

HARD RULES:
- PATTERN CAN MERELY CHANGE BASED ON REASONING ABOUT NATURAL LANGUAGE QUESTION.
- USE ONLY the given URIs.
- DO NOT invent, assume, or modify any URI.

EXAMPLE SCENARIOS:
{examples}

Question:
{Question}

PATTERNS AND URIs MUST BE SELECTED BASED ON THE DETAILS GIVEN WITH RESPECT TO THE QUESTION.

INPUTS:

SHOULD USE THIS PATTERN:
{Pattern}

ENTITY URIs:
{Entity_URIs}

RELATION URIs:
{Relation_URIs}

ADDITIONAL DATA FOR FINAL CORRECTION:

EXTERNAL TRIPLES:
{tripple}


OUTPUT FORMAT:
- Return ONLY valid JSON with the field "sparql".
"""

    evaluator_template = PromptTemplate(
        template=evaluator_prompt,
        input_variables=[
            "examples",
            "Question",
            "Pattern",
            "Entity_URIs",
            "Relation_URIs",
            "tripple",
            "current_query",
            "attempt_history",   # ← new
            "iteration",         # ← new
            "max_iterations"     # ← new
        ]
    )

    evaluator_llm = llm.with_structured_output(SPARQLQuery)

    # ---------------- DEBATE LOOP ---------------- #

    attempt_history = [current_query]   # ← track all queries seen so far

    for i in range(max_iterations):

        # Build readable history string for the prompt
        history_str = "\n".join(
            f"Attempt {idx + 1}:\n{q}" for idx, q in enumerate(attempt_history)
        )

        formatted_eval = evaluator_template.format(
            examples=text,
            Question=nl_query,
            Pattern=Pattern,
            Entity_URIs=entity,
            Relation_URIs=relation,
            tripple=tripple,
            current_query=current_query,
            attempt_history=history_str,   # ← injected
            iteration=i + 1,               # ← injected
            max_iterations=max_iterations  # ← injected
        )

        eval_messages = [
            SystemMessage(content="Evaluate and correct SPARQL queries."),
            HumanMessage(content=formatted_eval),
        ]

        evaluated = evaluator_llm.invoke(eval_messages)

        if evaluated.sparql.strip() == current_query.strip():
            break

        current_query = evaluated.sparql
        attempt_history.append(current_query)   # ← record every new query

    response = SPARQLQuery(sparql=current_query)

    return response.sparql

def validator(question:str,examples: list,uris:dict,pattern:str,global_dict:dict,llm):
  sparql = GENERATOR_AGENT3(question,examples,uris,pattern,global_dict,llm)
  return sparql

def validator2(nl_query: str,example:list, URIs: dict, Patterns: str, Global_Dict: dict, llm: ChatOpenAI):
  sparql = GENERATOR_AGENT2(nl_query,example, URIs, Patterns, Global_Dict, llm)
  return sparql

if __name__ == "__main__":
    pass





