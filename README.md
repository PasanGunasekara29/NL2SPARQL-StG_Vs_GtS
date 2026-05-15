# NL2SPARQL-StG_Vs_GtS
# Reasoning-Driven and Resource-Efficient NL2SPARQL using LLMs

## Overview

This repository presents a reasoning-driven framework for **Natural Language to SPARQL (NL2SPARQL)** conversion using **Large Language Models (LLMs)**. The framework focuses on generating accurate SPARQL queries from natural language questions while remaining computationally efficient for deployment in resource-constrained environments.

The system introduces two complementary reasoning workflows:

* **Generation-then-Selection (GtS)**
* **Selection-then-Generation (StG)**

Both approaches combine:

* Retrieval-Augmented Generation (RAG)
* LLM-as-a-Judge
* Iterative self-reflection
* Self-consistency inspired reasoning
* URI-aware prompting strategies

The framework is designed for heterogeneous Knowledge Graphs and consumer-grade LLM deployments.

---

## Key Features

✅ Dual reasoning workflows (GtS & StG)

✅ LLM-as-a-Judge for SPARQL validation and selection

✅ Self-reflection driven iterative refinement

✅ Retrieval-Augmented Generation (RAG)

✅ Reduced URI hallucination

✅ Supports domain-specific and general Knowledge Graphs

✅ Resource-efficient deployment with small/consumer LLMs

✅ Strong generalization across unseen query patterns

---

## Architecture

### Generation-then-Selection (GtS)

1. Retrieve relevant NL–SPARQL examples using RAG
2. Build multiple prompt templates
3. Generate candidate SPARQL queries
4. Apply iterative self-reflection
5. Use LLM-as-a-Judge to select the best candidate

### Selection-then-Generation (StG)

1. Retrieve relevant examples
2. Generalize SPARQL templates
3. Use Judge module to select optimal structure
4. Generate SPARQL query
5. Refine using iterative self-reflection

---

## Project Structure

```text
.
├── data/
│   ├── standard_datasets/
│   └── custom_datasets/
│
├── models/
│   └── LLM configurations
│
├── rag/
│   └── Retrieval pipeline
│
├── judge/
│   └── LLM-as-a-Judge modules
│
├── self_reflection/
│   └── Iterative refinement logic
│
├── orchestrator/
│   └── GtS and StG workflows
│
├── evaluation/
│   └── Metrics and benchmarks
│
└── main.py
```

---

## Datasets

The framework supports multiple benchmark datasets:

| Dataset                | Knowledge Graph  |
| ---------------------- | ---------------- |
| LC-QuAD 2.0            | Wikidata         |
| VQuAnDa                | DBpedia          |
| QALD-9                 | DBpedia          |
| QALD-9 Plus            | Wikidata         |
| Custom Weather Dataset | Climate Ontology |

---

## Installation

```bash
git clone <repository-url>

cd repository-name

pip install -r requirements.txt
```

---

## Run

Example:

```bash
python main.py
```

Input:

```text
What is the average temperature of New York City?
```

Output:

```sparql
SELECT ?avgTemp
WHERE {
   ...
}
```

---

## Evaluation Metrics

Performance is evaluated using:

* BLEU
* F1
* Ensemble BLEU
* Ensemble F1
* Execution F1

---

## Supported Models

Examples:

* Qwen2.5
* Llama 3.2
* Phi-4 Mini
* Granite

Models are optimized for local deployment and consumer-grade GPUs.

---

## Citation

If you use this work in research:

```bibtex
@article{nl2sparql_reasoning,
 title={Reasoning-Driven and Resource-Efficient SPARQL Query Generation Using Large Language Models}
}
```

---

## License

MIT License

---

## Acknowledgments

Built for scalable and efficient Knowledge Graph Question Answering using reasoning-driven Large Language Models.
