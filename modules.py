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
  uris = URI_Extraction(kb, question)
  return uris

def GENERATOR_AGENT(
        nl_query: str,
        examples: list,
        SPARQL_LIST: list,
        URIS: dict,
        Global_Dict: dict,
        context: str,
        llm,
        llm2,
        max_iterations: int = 3
) -> str:
    if URIS:
        entity = URIS.get("entity")
        relation = URIS.get("relation")
    else:
        entity = None
        relation = None

    text = ""
    for items in examples:
        text += items
        text += "\n\n"

    SPARQL = ""
    sparql_list = SPARQL_CONTEXT(SPARQL_LIST, SPARQL)

    # --- KEY FIX: Number the patterns so the LLM picks by index ---
    numbered_patterns = "\n\n".join(
        f"[{i}] {p}" for i, p in enumerate(SPARQL_LIST)
    )

    prompt5 = """You are an expert in generalized SPARQL pattern selection from a given list of generalized SPARQL queries that align with given NL question and context.
TASK:
Select ONLY THE CORRECT GENERALIZED SPARQL pattern from the provided list of SPARQL patterns based on the natural language question, given URIs and context.

You are ONLY allowed to pick ONE SPARQL pattern from the provided list that best matches the given details.

Use the index of the  pattern from the numbered list below that best matches the question, URIs, and context to select a one.

The selected SPARQL pattern must be the one that can generate the correct SPARQL query structure for the given context.

STRICT RULES:
1. The index of chosen SPARQL pattern MUST be exactly one of the patterns from the provided list.
2. Do NOT modify, rewrite, summarize, or generate a new pattern.
3. Do NOT explain your reasoning.
4. Output ONLY the final selected SPARQL pattern.
5. Return ONLY valid index for the most aligned "pattern".
6. Picking a generalize SPARQL pattern from a given generalize SPARQWL patern is must.
7. The index MUST correspond to one of the numbered patterns in the list.

SELECTION GUIDELINES:
- Use Entity URIs to identify the subject or object.
- Use Relation URIs to identify the predicate.
- Use the natural language question to determine the query intent.
- Match the structure of the question to the structure of the SPARQL pattern.
- Prefer the most specific matching pattern.
- If multiple patterns seem similar, choose the one that best satisfies the required relation and entity roles.
- You should pick a generalize SPARQL from the given List of SPARQL generalize Patterns.

Context:
{context}

Examples fo Nl to SPARQL conversion:
{examples}

Relevant Entity URIs:
{entity_uris}

Relevant Relation URIs:
{relation_uris}

External URIs:
{external_uris}

Given Question:
{question}

Picked pattern from List of SPARQL Patterns:

List of SPARQL Patterns:
{PROMPT_LIST}


Index of picked pattern from above list:
{{<<Selected SPARQL Pattern index}}
"""

    PROMPT = PromptTemplate(
        template=prompt5,
        input_variables=[
            "context", "examples", "entity_uris",
            "relation_uris", "external_uris", "PROMPT_LIST", "question"
        ]
    )

    formatted_prompt = PROMPT.format(
        context=context,
        examples=text,
        entity_uris=entity,
        relation_uris=relation,
        external_uris=Global_Dict,
        PROMPT_LIST=numbered_patterns,
        question=nl_query
    )

    messages = [
        SystemMessage(content="You select the correct SPARQL pattern index from a numbered list."),
        HumanMessage(content=formatted_prompt)
    ]

    llm_select = llm.with_structured_output(PatternIndexSelect)  # field: index: int
    generated = llm_select.invoke(messages)
    current_index = _clamp_index(generated.index, SPARQL_LIST)

    # --- Evaluator loop: also index-based ---
    evaluator_prompt = prompt5 + """
CURRENTLY SELECTED INDEX: {current_index}

TASK:
Verify whether the selected index with respect to genearlized SPARQL pattern is the correct one from the provided list of SPARQL patterns.

STRICT RULES:
1. The returned index of SPARQL pattern MUST be exactly one of the patterns from the provided list.
2. Do NOT modify, rewrite, or generate a new indexes.
3. If the current pattern is correct, return it unchanged.
4. If the current pattern is incorrect, select the correct SPARQL pattern from the given list.
5. Do NOT explain your reasoning.
6. Return ONLY valid index for the "pattern".
7. You should pick an index for generalize SPARQL from the list 


{{SPARQL QUERY}}.



Return ONLY: {{"index": <integer>}}
"""

    evaluator_template = PromptTemplate(
        template=evaluator_prompt,
        input_variables=[
            "context", "examples", "entity_uris", "relation_uris",
            "external_uris", "PROMPT_LIST", "question", "current_index"
        ]
    )

    for _ in range(max_iterations):
        formatted_eval = evaluator_template.format(
            context=context,
            examples=text,
            entity_uris=entity,
            relation_uris=relation,
            external_uris=Global_Dict,
            PROMPT_LIST=numbered_patterns,
            question=nl_query,
            current_index=current_index
        )

        llm_select2 = llm2.with_structured_output(PatternIndexSelect)
        eval_messages = [
            SystemMessage(content="Evaluate the selected SPARQL pattern index."),
            HumanMessage(content=formatted_eval)
        ]

        evaluated = llm_select2.invoke(eval_messages)
        new_index = _clamp_index(evaluated.index, SPARQL_LIST)

        if new_index == current_index:
            break

        current_index = new_index

    # --- Guaranteed list member returned ---
    return SPARQL_LIST[current_index]


def _clamp_index(index: int, sparql_list: list) -> int:
    """Ensures index is always within valid range."""
    return max(0, min(index, len(sparql_list) - 1))

class PatternIndexSelect(BaseModel):
    index: int  # replaces PatternSelect(prompt: str)


def GENERATOR_AGENT3(
    nl_query: str,
    examples: list,
    URIs: dict,
    Pattern: str,
    Global_Dict: dict,
    llm: ChatOpenAI,
    llm2: ChatOpenAI,
    max_iterations: int = 5
) -> str:

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

MUST USE ONLY THESE provided ENTITY URIs:
{Entity_URIs}

MUST USE ONLY THESE provided RELATION URIs:
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

    evaluator_prompt = """You are an assistant that evaluate SPARQL queries based on a given natural language question, predefined query patterns, and provided URIs.

CURRENT QUERY:
{current_query}

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

MUST USE ONLY THESE provided ENTITY URIs:
{Entity_URIs}

MUST USE ONLY THESE provided RELATION URIs:
{Relation_URIs}

ADDITIONAL DATA FOR FINAL CORRECTION:

EXTERNAL TRIPLES:
{tripple}


{{SPARQL QUERY}}

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
            "current_query"
        ]
    )

    evaluator_llm = llm2.with_structured_output(SPARQLQuery)

    # ---------------- DEBATE LOOP ---------------- #

    for _ in range(max_iterations):

        formatted_eval = evaluator_template.format(
            examples=text,
            Question=nl_query,
            Pattern=Pattern,
            Entity_URIs=entity,
            Relation_URIs=relation,
            tripple=tripple,
            current_query=current_query
        )

        eval_messages = [
            SystemMessage(content="Evaluate and correct SPARQL queries."),
            HumanMessage(content=formatted_eval),
        ]

        evaluated = evaluator_llm.invoke(eval_messages)

        if evaluated.sparql.strip() == current_query.strip():
            break

        current_query = evaluated.sparql

    response = SPARQLQuery(sparql=current_query)

    return response.sparql


if __name__ == "__main__":
    pass