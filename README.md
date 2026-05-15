# NL2SPARQL-StG_Vs_GtS

# Reasoning-Driven and Resource-Efficient NL2SPARQL using LLMs

<p align="center">
Reasoning-driven NL2SPARQL framework using LLMs with GtS/StG workflows, RAG, self-reflection, and LLM-as-a-Judge for scalable SPARQL generation.
</p>

---

## Overview

This repository presents a reasoning-driven framework for **Natural Language to SPARQL (NL2SPARQL)** conversion using **Large Language Models (LLMs)**. The framework enables efficient and scalable SPARQL generation while reducing computational overhead for deployment in resource-constrained environments.

The proposed system introduces two complementary reasoning workflows:

- **Generation-then-Selection (GtS)**
- **Selection-then-Generation (StG)**

The framework integrates:

- Retrieval-Augmented Generation (RAG)
- LLM-as-a-Judge
- Iterative self-reflection
- Self-consistency inspired reasoning
- URI-aware prompting

---

## Key Features

✅ Dual reasoning workflows (GtS & StG)

✅ LLM-as-a-Judge based query validation

✅ Iterative self-reflection mechanism

✅ Retrieval-Augmented Generation (RAG)

✅ Reduced URI hallucination

✅ Resource-efficient deployment

✅ Generalization across heterogeneous knowledge graphs

---

# Architecture

## Generation-then-Selection (GtS)

<p align="center">
  <img src="assets/GtS_Approach.png" width="900" alt="GtS Architecture">
</p>

The GtS workflow generates multiple candidate SPARQL queries through template-guided reasoning paths and selects the most contextually aligned output using an LLM Judge module.

Workflow:

1. Retrieve relevant NL–SPARQL examples
2. Generate prompt templates
3. Produce multiple candidate queries
4. Apply iterative self-reflection
5. Select the best query using LLM-as-a-Judge

---

## Selection-then-Generation (StG)

<p align="center">
  <img src="assets/StG_Approach.png" width="900" alt="StG Architecture">
</p>

The StG workflow first selects an optimal generalized SPARQL pattern and subsequently generates a query refined through iterative reasoning.

Workflow:

1. Retrieve examples
2. Generate generalized templates
3. Select template using Judge module
4. Generate SPARQL query
5. Refine using self-reflection

---

## Project Structure

```text
.
├── assets/
│   ├── GtS_Approach.png
│   └── StG_Approach.png
│
├── data/
├── rag/
├── judge/
├── self_reflection/
├── orchestrator/
├── evaluation/
└── main.py
