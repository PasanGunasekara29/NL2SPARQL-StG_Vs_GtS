"""
sparql_eval_utils.py
====================
Utilities for:
  1. Text normalization & evaluation metrics  (BLEU, F1, Exact Match)
  2. SPARQL generalization & URI extraction   (DBpedia + Wikidata)
  3. SPARQL validation against live endpoints
  4. Context builders for retrieval-augmented generation
  5. Wikidata label resolution (live API)
  6. DBpedia query execution & answer-level accuracy
  7. Pydantic output models for structured LLM responses
"""

# ─────────────────────────────────────────────────────────────────────────────
# Standard-library & third-party imports
# ─────────────────────────────────────────────────────────────────────────────
import re
import string
import json
from collections import Counter

import time

import requests
import nltk
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sacrebleu.metrics import BLEU as SacreBLEU
from pydantic import BaseModel, Field
from SPARQLWrapper import SPARQLWrapper, JSON
from urllib.error import HTTPError, URLError
from NER_Agent import entity_linking, relation_linking

nltk.download("punkt", quiet=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  TEXT NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace."""
    if text is None:
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ─────────────────────────────────────────────────────────────────────────────
# 3.  SPARQL GENERALIZATION  (replace concrete URIs with placeholders)
# ─────────────────────────────────────────────────────────────────────────────

_PLACEHOLDER = "{{Replace with URI}}"


def generalize_sparql_dbpedia(query: str) -> str:
    """
    Replace all DBpedia resource/ontology/property URIs (full or prefixed)
    with a generic placeholder, leaving PREFIX declarations untouched.
    """
    def _replace_full(match):
        # Don't touch URIs that appear after a PREFIX keyword
        return _PLACEHOLDER if "PREFIX" not in query[:match.start()] else match.group()

    query = re.sub(
        r"<http://dbpedia\.org/(resource|ontology|property)/[^>]+>",
        _replace_full,
        query,
    )
    query = re.sub(r"\b(dbo|dbp|dbr|res):[A-Za-z0-9_\-]+", _PLACEHOLDER, query)
    return query


def generalize_sparql_dbpedia_typed(query: str) -> str:
    """
    Replace DBpedia URIs with typed placeholders that distinguish
    rdf:type relations, ontology/property relations, and resource entities.
    Leaves PREFIX declarations untouched.
    """
    query = query.strip()
    query = re.sub(r"<http[^>]*rdf-syntax-ns#type[^>]*>",    "{{Replace With Relation URIs}}", query)
    query = re.sub(r"<http[^>]*(ontology|property)/[^>]*>",  "{{Replace With Relation URIs}}", query)
    query = re.sub(r"<http[^>]*resource/[^>]*>",             "{{Replace With Entity URIs}}",   query)
    query = re.sub(r"\s+", " ", query).strip()
    return query


def replace_prefixed_uris(sparql_query: str,
                           placeholder: str = "{{Replace with URI}}",
                           skip_prefixes: tuple = ("xsd:",)) -> str:
    """
    Replace all prefixed URIs (e.g. sosa:resultTime, qudt:numericValue) with
    a placeholder, while keeping PREFIX declaration lines and any listed
    ``skip_prefixes`` (e.g. xsd: datatypes) intact.
    """
    prefixed_uri_pattern = re.compile(r"\b[a-zA-Z_][\w\-]*:[\w\-]+\b")
    lines = []
    for line in sparql_query.splitlines():
        if line.strip().upper().startswith("PREFIX"):
            lines.append(line)
            continue
        def _replace(match):
            token = match.group(0)
            return token if any(token.startswith(p) for p in skip_prefixes) else placeholder
        lines.append(prefixed_uri_pattern.sub(_replace, line))
    return "\n".join(lines)


def replace_uris_with_placeholder(sparql_text: str,
                                   placeholder: str = "{{Replace with URIs}}") -> str:
    """Replace every angle-bracket URI  (<...>)  with a placeholder."""
    return re.sub(r"<[^>]*>", placeholder, sparql_text)


def extract_uris_library(sparql_query: str) -> dict:
    """
    Extract DBpedia URIs from a SPARQL query and classify as entity or relation.

    Returns
    -------
    {'entity': [(label, uri), ...], 'relation': {label: uri, ...}}
    """
    uris     = re.findall(r"<(http[^>]+)>", sparql_query)
    uri_dict = {"entity": [], "relation": {}}

    for uri in uris:
        label = uri.split("/")[-1].replace("_", " ")
        if "dbpedia.org/resource/" in uri:
            uri_dict["entity"].append((label, uri))
        elif "dbpedia.org/ontology/" in uri or "dbpedia.org/property/" in uri:
            uri_dict["relation"][label] = uri

    return uri_dict


def context_with_uris(retrieve: list[dict]) -> str:
    """
    Numbered list of URI library dicts, one per retrieved example.
    Useful for injecting entity/relation context into prompts.
    """
    lines = []
    for i, item in enumerate(retrieve):
        lines.append(f"library {i} : {extract_uris_library(item['sparql_query'])}")
    return "\n".join(lines)


def generalize_sparql_wikidata(query: str) -> str:
    """Replace all Wikidata entity (wd:Qxxx) and property (wdt:Pxxx) URIs."""
    query = re.sub(r"wdt:P\d+", "{{Replace With Relation URI}}", query)
    query = re.sub(r"wd:Q\d+",  "{{Replace With Entity URI}}",   query)
    query = re.sub(r"\s+", " ", query).strip()
    return query


def replace_uris_wikidata_generic(query: str) -> str:
    """Replace any wd/wdt/p/ps/pq prefixed token with a single placeholder."""
    return re.sub(r"\b(?:wd|wdt|p|ps|pq):[A-Za-z0-9]+\b",
                  "{{Replace with URIs}}", query)

def replace_sparql_uris_QALD9(query):

    placeholder = "{{Replace with URI}}"

    # replace full DBpedia URIs except those in PREFIX
    def replace_full(match):
        uri = match.group()
        if "PREFIX" in query[:match.start()]:
            return uri
        return placeholder

    query = re.sub(
        r"<http://dbpedia\.org/(resource|ontology|property)/[^>]+>",
        replace_full,
        query
    )

    # replace prefixed URIs
    query = re.sub(
        r"\b(dbo|dbp|dbr|res):[A-Za-z0-9_\-]+",
        placeholder,
        query
    )

    return query

def replace_entity_relation_uris_custom(sparql_query: str) -> str:
    """
    Replace all entity and relation URIs with {{Rplace with URis}}
    while keeping PREFIX declarations unchanged.
    """

    lines = sparql_query.splitlines()
    processed_lines = []

    # Regex to match prefixed names like sosa:resultTime
    prefixed_uri_pattern = re.compile(r'\b[a-zA-Z_][\w\-]*:[\w\-]+\b')

    for line in lines:
        stripped = line.strip()

        # Keep PREFIX lines exactly as they are
        if stripped.upper().startswith("PREFIX"):
            processed_lines.append(line)
            continue

        # Replace prefixed URIs in other lines
        def replacer(match):
            token = match.group(0)

            # Do not replace xsd:date (datatype)
            if token.startswith("xsd:"):
                return token

            # Replace all other entity/relation URIs
            return PLACEHOLDER

        new_line = prefixed_uri_pattern.sub(replacer, line)
        processed_lines.append(new_line)

    return "\n".join(processed_lines)
# ─────────────────────────────────────────────────────────────────────────────
# 4.  URI EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_uris_dbpedia(sparql_query: str) -> tuple[dict, dict]:
    """
    Extract DBpedia entity and relation URIs from a SPARQL query.

    Returns
    -------
    (entities, relations)  — both are {label: uri} dicts
    """
    uris      = re.findall(r"<(http[^>]+)>", str(sparql_query))
    entities  = {}
    relations = {}

    for uri in uris:
        if "dbpedia.org" not in uri:
            continue
        tag = uri.split("/")[-1].replace("_", " ")
        if "/resource/" in uri:
            entities[tag]  = uri
        else:
            relations[tag] = uri

    return entities, relations


def extract_uris_wikidata(sparql_query: str) -> dict:
    """
    Extract Wikidata entity (wd:Qxxx) and property (wdt:Pxxx) URIs.

    Returns
    -------
    {'entity': [(id, uri), ...], 'relation': {id: uri, ...}}
    """
    result = {"entity": [], "relation": {}}

    for e in re.findall(r"wd:Q\d+", sparql_query):
        eid = e.split(":")[1]
        if (eid, e) not in result["entity"]:
            result["entity"].append((eid, e))

    for r in re.findall(r"wdt:P\d+", sparql_query):
        rid = r.split(":")[1]
        result["relation"][rid] = r

    return result


def extract_wikidata_ids_with_tags(data: dict) -> dict:
    """
    Extract Wikidata Q/P IDs from a URI dict and attach minimal semantic tags.

    Input  : {'entity': [(label, uri), ...], 'relation': [uri, ...]}
    Output : {'entity': [(label, Qxxx), ...], 'relation': [(Pxxx, tag), ...]}
    """
    _relation_tags = {"P155": "follows", "P156": "followed by"}
    output = {"entity": [], "relation": []}

    for label, uri in data.get("entity", []):
        if uri:
            m = re.search(r"(Q\d+)", uri)
            output["entity"].append((label, m.group(1) if m else None))
        else:
            output["entity"].append((label, None))

    for uri in data.get("relation", []):
        m = re.search(r"(P\d+)", uri)
        if m:
            pid = m.group(1)
            output["relation"].append((pid, _relation_tags.get(pid, "relation")))

    return output


# ──triple-level extraction ──────────────────────────────────────────


def uri_to_label(uri):
    if uri.startswith("<") and uri.endswith(">"):
        uri = uri[1:-1]
    return uri.split("/")[-1].replace("_", " ")

def extract_triples(sparql):
    where = re.search(r"\{(.*)\}", sparql, re.DOTALL)
    if not where:
        return []
    block = where.group(1)
    pattern = r"(<[^>]+>|[?]\w+)\s+(<[^>]+>)\s+(<[^>]+>|[?]\w+)"


def _uri_to_label(uri: str) -> str:
    uri = uri.strip("<>")
    return uri.split("/")[-1].replace("_", " ")


def extract_triples(sparql: str) -> list[tuple]:
    """Return (subject, predicate, object) tuples from the WHERE clause."""
    where = re.search(r"\{(.*)\}", sparql, re.DOTALL)
    if not where:
        return []
    pattern = r"(<[^>]+>|[?]\w+)\s+(<[^>]+>)\s+(<[^>]+>|[?]\w+)"
    return re.findall(pattern, where.group(1))

def extract_entities(triples):
    entities = set()
    for s, p, o in triples:
        if s.startswith("<"):
            entities.add(s)
        if o.startswith("<"):
            entities.add(o)
    # convert to (label, uri)
    return [(uri_to_label(e), e[1:-1]) for e in entities]

def extract_relations(triples):
    relations = set()
    for _, p, _ in triples:
        relations.add(p)
    # convert to label: uri
    return {uri_to_label(r): r[1:-1] for r in relations}

def build_global_uri_dict(data: list[dict]) -> dict:
    """
    Build a merged entity/relation/triple dict from a list of
    {'sparql_query': ...} records.
    """
    all_entities  = {}
    all_relations = {}
    all_triples   = []

    for item in data:
        triples = extract_triples(item["sparql_query"])
        all_triples.extend(triples)

        for s, p, o in triples:
            for token in (s, o):
                if token.startswith("<"):
                    uri   = token.strip("<>")
                    label = _uri_to_label(token)
                    all_entities[uri] = label
            if p.startswith("<"):
                uri   = p.strip("<>")
                label = _uri_to_label(p)
                all_relations[uri] = label

    return {
        "entity":   [(label, uri) for uri, label in all_entities.items()],
        "relation": {label: uri  for uri, label in all_relations.items()},
        "triples":  all_triples,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  WIKIDATA LABEL RESOLUTION (live API)
# ─────────────────────────────────────────────────────────────────────────────

def resolve_wikidata_labels(data: dict, lang: str = "en") -> dict:
    """
    Resolve Wikidata Q/P IDs to human-readable labels via the MediaWiki API.

    Input  : {'entity': [(label, uri), ...], 'relation': [uri, ...]}
    Output : {'entity': [(real_label, Qxxx), ...],
              'relation': [(real_label, Pxxx), ...]}
    """
    qids, pids = set(), set()

    for _, uri in data.get("entity", []):
        if uri:
            m = re.search(r"(Q\d+)", uri)
            if m:
                qids.add(m.group(1))

    for uri in data.get("relation", []):
        m = re.search(r"(P\d+)", uri)
        if m:
            pids.add(m.group(1))

    all_ids = list(qids | pids)
    if not all_ids:
        return {"entity": [], "relation": []}

    url     = "https://www.wikidata.org/w/api.php"
    headers = {"User-Agent": "AgentSPARQL/1.0 (https://example.com; email@example.com)"}
    params  = {
        "action":    "wbgetentities",
        "ids":       "|".join(all_ids),
        "props":     "labels",
        "languages": lang,
        "format":    "json",
    }

    resp = requests.get(url, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    api_data = resp.json().get("entities", {})

    def _label(wid):
        return api_data.get(wid, {}).get("labels", {}).get(lang, {}).get("value", "unknown")

    return {
        "entity":   [(_label(q), q) for q in qids],
        "relation": [(_label(p), p) for p in pids],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  SPARQL VALIDATION (live endpoints)
# ─────────────────────────────────────────────────────────────────────────────

_ENDPOINTS = {
    "wikidata": "https://query.wikidata.org/sparql",
    "dbpedia":  "https://dbpedia.org/sparql",
}


def check_sparql_validity(query: str, endpoint: str = "wikidata") -> tuple[bool, str]:
    """
    Validate a SPARQL query against the Wikidata or DBpedia endpoint.

    Returns
    -------
    (is_valid, message)
    """
    url = _ENDPOINTS.get(endpoint.lower())
    if not url:
        return False, "Invalid endpoint. Choose 'wikidata' or 'dbpedia'."

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; sparql-validator)",
        "Accept":     "application/sparql-results+json",
    }
    try:
        resp = requests.get(url, params={"query": query},
                            headers=headers, timeout=15)
        if resp.status_code == 200:
            return True, "Query is syntactically valid."
        return False, f"Endpoint error {resp.status_code}: {resp.text}"
    except Exception as exc:
        return False, f"Request failed: {exc}"


def parse_sparql_from_response(response: str) -> str:
    """
    Extract a raw SPARQL query from an LLM response that may use
    code fences (```...```) or a '[sparql]:' prefix.
    Returns 'no sparql' if nothing is found.
    """
    if "```" in response:
        return response.split("```")[1].strip()
    if "[sparql]:" in response:
        return response.split("[sparql]:")[1].strip()
    return "no sparql"


def validate_sparql(response: str, kb: str) -> tuple[bool, str]:
    """
    Parse + validate a SPARQL query extracted from an LLM response.

    Parameters
    ----------
    response : raw LLM output string
    kb       : 'dbpedia' or 'wikidata'

    Returns
    -------
    (is_valid, extracted_query)
    """
    query = parse_sparql_from_response(response)
    if query == "no sparql":
        return False, query

    kb = kb.lower()
    if kb in ("dbpedia", "wikidata"):
        is_valid, _ = check_sparql_validity(query, kb)
        return is_valid, query

    return True, query   # unknown KB — assume valid


# ─────────────────────────────────────────────────────────────────────────────
# 7.  CONTEXT BUILDERS  (for RAG / few-shot prompting)
# ─────────────────────────────────────────────────────────────────────────────
def context(retrieve :list):
  text= ""
  for items in range(len(retrieve)):
    text += f"{retrieve[items].get('english_text')} to SPARQL is {retrieve[items].get('sparql_query')} \n"
  return text

def context_simple(retrieve: list[dict]) -> str:
    """'<question> to SPARQL is <query>' — one line per example."""
    return "\n".join(
        f"{item['english_text']} to SPARQL is {item['sparql_query']}"
        for item in retrieve
    )


def context_with_patterns(retrieve: list[dict]) -> str:
    """Numbered list of generalized SPARQL patterns."""
    lines = []
    for i, item in enumerate(retrieve, 1):
        pattern = generalize_sparql_wikidata(item["sparql_query"])
        lines.append(f'--> Pattern {i} : "{pattern}"')
    return "\n".join(lines)


def context_dynamic(retrieve: list[dict]) -> list[str]:
    """
    Rich per-example context blocks with question, note, and full SPARQL.
    """
    contexts = []
    for item in retrieve:
        block  = f"[QUESTION] : {item['english_text']}\n"
        block += "Have converted to SPARQL queries using the following Relation and Entity URIs.\n"
        block += f"[SPARQL] : {item['sparql_query']}"
        contexts.append(block)
    return contexts


def sparql_context_with_selection(sparql_list: list[str], selected: str) -> str:
    """Concatenate a candidate list with the selected query at the bottom."""
    return "\n".join(sparql_list) + "\n\nSelected:\n" + str(selected)


def URI_Extraction(kb:str, question:str):
  entity_extraction = entity_linking(kb, question)
  relation_extraction = relation_linking(kb, question)
  return {"entity":entity_extraction, "relation":relation_extraction}

def CONTEXT_CREATOR_DYNAMIC(retrieve):
  contexts = []
  for items in retrieve:
    uris = URI_Extraction("dbpedia", items.get('english_text'))#extract_entities_relations_sparql(items.get("sparql_query"))#URI_Extraction("dbpedia", items.get('english_text'))#extract_entities_relations_sparql(items.get("sparql_query"))#URI_Extraction("dbpedia", items.get('english_text'))#extract_entities_relations_sparql(items.get("sparql_query"))#extract_entities_relations_sparql(items.get("sparql_query"))#URI_Extraction("dbpedia", items.get('english_text'))#extract_entities_relations_sparql(items.get("sparql_query"))
    text = ""
    text += f"[QUESTION] : {items.get('english_text')}  \n"
    text += f"Have converted to SPARQL queries using following Relation and Entity URIs.\n"
    text += f"[ENTITY URIs] : {uris["entity"]}"#{uris[0]}{uris[0]}  \n{uris["entity"]} 
    text += f"[RELATION URIs] : {uris["relation"]} \n"#{uris[1]}  \nuris["relation"]
    text += f"[SPARQL] : {items.get("sparql_query")}"
    contexts.append(text)
  return contexts
# ─────────────────────────────────────────────────────────────────────────────
# 6.  DBPEDIA QUERY EXECUTION & ANSWER-LEVEL ACCURACY
# ─────────────────────────────────────────────────────────────────────────────

def query_dbpedia(
    sparql_query: str,
    timeout: int = 20,
    max_retries: int = 5,
    backoff_factor: float = 2.0,
) -> list:
    """
    Execute a SPARQL query against DBpedia with exponential-backoff retries.

    Returns a flat list of answer values, or an empty list if all retries fail.
    """
    sparql = SPARQLWrapper("https://dbpedia.org/sparql")
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(timeout)

    attempt, wait_time = 0, 1.0

    while attempt < max_retries:
        try:
            results = sparql.query().convert()
            return [
                row[var]["value"]
                for row in results["results"]["bindings"]
                for var in row
            ]
        except (HTTPError, URLError, Exception) as exc:
            attempt += 1
            if attempt >= max_retries:
                print("[ERROR] SPARQL failed after all retries")
                print("Query:", sparql_query)
                print("Last error:", exc)
                return []
            print(f"[WARN] SPARQL error (attempt {attempt}/{max_retries}): {exc}")
            time.sleep(wait_time)
            wait_time *= backoff_factor


def sparql_accuracy(reference_query: str, candidate_query: str) -> dict:
    """
    Compute answer-level accuracy by executing both queries on DBpedia.

    Rules
    -----
    - Both empty  →  accuracy = 1.0  (NULL == NULL)
    - GT empty, candidate non-empty  →  accuracy = 0.0
    - Otherwise  →  |GT ∩ Candidate| / |GT|

    Returns
    -------
    dict with keys: accuracy, reference_answers, candidate_answers, overlap_answers
    """
    ref_answers  = set(query_dbpedia(reference_query))
    cand_answers = set(query_dbpedia(candidate_query))

    if not ref_answers and not cand_answers:
        return {"accuracy": 1.0, "reference_answers": ref_answers,
                "candidate_answers": cand_answers, "overlap_answers": set()}

    if not ref_answers:
        return {"accuracy": 0.0, "reference_answers": ref_answers,
                "candidate_answers": cand_answers, "overlap_answers": set()}

    overlap  = ref_answers & cand_answers
    accuracy = len(overlap) / len(ref_answers)
    return {
        "accuracy":           accuracy,
        "reference_answers":  ref_answers,
        "candidate_answers":  cand_answers,
        "overlap_answers":    overlap,
    }

def build_global_dict(data):
    all_entities = {}
    all_relations = {}
    all_triples = []

    for item in data:
        triples = extract_triples(item["sparql_query"])
        all_triples.extend(triples)

        # merge entities
        for label, uri in extract_entities(triples):
            all_entities[uri] = label  # use URI as key, keep label

        # merge relations
        for label, uri in extract_relations(triples).items():
            all_relations[uri] = label

    # convert entity dict back to required list form
    entity_list = [(label, uri) for uri, label in all_entities.items()]

    # convert relation dict back to required dict form
    relation_dict = {label: uri for uri, label in all_relations.items()}

    return {
        "entity": entity_list,
        "relation": relation_dict,
        "triples": all_triples
    }
    
def SPARQL_CONTEXT(sparql_list: list, sparql: str):
    text = ""
    for q in sparql_list:
        text += q + "\n"
    text += "\nSelected:\n" + str(sparql)
    return text

# ─────────────────────────────────────────────────────────────────────────────
# 7.  PYDANTIC OUTPUT MODELS  (structured LLM responses)
# ─────────────────────────────────────────────────────────────────────────────

class SPARQLQuery(BaseModel):
    """LLM must return a single valid SPARQL query."""
    sparql: str = Field(
        description="A valid SPARQL query ONLY. No explanation, no markdown."
    )


class SPARQLQuerySelect(BaseModel):
    """LLM must select one SPARQL query from a provided list."""
    sparql: str = Field(
        description="A valid SPARQL query selected from the given list."
    )


class PromptSelect(BaseModel):
    """LLM must select one prompt from a provided list."""
    prompt: str = Field(
        description="A valid prompt selected from the given prompt list."
    )


class PatternSelect(BaseModel):
    """LLM must select one generalized SPARQL pattern from a provided list."""
    prompt: str = Field(
        description="A generalized SPARQL pattern selected from the given list."
    )
    
    


if __name__ == "__main__":
    pass