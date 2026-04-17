# Knowledge Graph

A Python tool that builds a deterministic knowledge graph from lease and property CSV data, enriches it optionally with LLM-generated insights via Ollama, and supports natural-language querying, chatbot interaction, and multi-format export.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Graph Model](#graph-model)
- [Requirements](#requirements)
- [Input Data](#input-data)
- [Usage](#usage)
- [Module Reference](#module-reference)
- [Output Files](#output-files)
- [Query Examples](#query-examples)

---

## Overview

The pipeline runs in three stages:

1. **Parse** — Reads lease and property records from CSV files using `data_utils.py`.
2. **Build** — Constructs a directed multigraph (`networkx.MultiDiGraph`) in `graph_builder.py`, linking leases to properties, cities, states, markets, and usage types.
3. **Enrich & Export** — Optionally calls a local Ollama LLM for business insights, then exports the graph to JSON, GraphML, and PNG via `exporters.py`.

Querying works in two modes:

- **Deterministic** — Pure graph traversal for market-based and property-area-based lookups. No LLM required.
- **LLM-assisted** — Passes relevant graph context to Ollama for natural-language answers and multi-turn chat.

If Ollama is not running the tool silently falls back to deterministic lexical search for all query and chat operations.

---

## Architecture

```
CSV files
    │
    ▼
data_utils.py  ──►  graph_builder.py  ──►  NetworkX MultiDiGraph
                         │
                         ├──►  ollama_client.py  (optional enrichment)
                         │
                         ▼
                    query_engine.py  ──►  answers / chat responses
                         │
                         ▼
                    exporters.py  ──►  .json / .graphml / .png / nlp report
                         │
                         ▼
                    knowledge_graph.py  (CLI orchestrator)
```

---

## Project Structure

```
knowledge_graph/
├── knowledge_graph.py   # CLI entrypoint and pipeline orchestration
├── graph_builder.py     # Graph construction and Ollama insight enrichment
├── query_engine.py      # Lexical search, deterministic traversal, QA, and chat
├── exporters.py         # JSON, GraphML, PNG, and NLP report export
├── ollama_client.py     # Stdlib HTTP client for the Ollama REST API
├── data_utils.py        # CSV parsing, type coercion, and text normalization
├── deps.py              # Lazy networkx loader
└── input/
    ├── lease_data.csv
    └── property_data.csv
```

---

## Graph Model

### Node Types

| Label | Description | Key Attributes |
|---|---|---|
| `Lease` | A single lease record | `lease_id`, `total_leased_space`, `asking_rent_monthly`, `base_rent_monthly`, `start_date`, `end_date`, `signed_date`, `arrangement_status`, `lease_activity_type`, `usage_type` |
| `Property` | A physical property | `property_id`, `address`, `city`, `state`, `postal_code`, `net_rentable_area`, `total_gross_area`, `last_sale_price`, `usage_type`, `class_type` |
| `City` | City derived from property data | `name` |
| `State` | State derived from property data | `code` |
| `Market` | Market segment of a property | `name` |
| `UsageType` | Property or lease usage classification | `name` |
| `Insight` | LLM-generated business insight (optional) | `text` |
| `InsightCollection` | Hub node that groups all insight nodes | — |

### Edge Types

| Relation | From → To | Description |
|---|---|---|
| `HAS_LEASED_PROPERTY` | `Lease` → `Property` | Links a lease to its property |
| `LOCATED_IN_CITY` | `Property` → `City` | Property's city |
| `LOCATED_IN_STATE` | `Property` → `State` | Property's state |
| `IN_MARKET` | `Property` → `Market` | Property's market segment |
| `LEASE_USAGE` | `Lease` → `UsageType` | Usage type sourced from the lease record |
| `PROPERTY_USAGE` | `Property` → `UsageType` | Usage type sourced from the property record |
| `HAS_INSIGHT` | `InsightCollection` → `Insight` | Connects the hub to each individual insight node |

---

## Requirements

| Package | Purpose | Required |
|---|---|---|
| `networkx` | Graph construction, traversal, and GraphML export | Yes |
| `matplotlib` | PNG image rendering | Optional |
| Ollama (local server) | LLM enrichment, QA, and chat | Optional |

Install required packages:

```bash
pip install networkx
pip install matplotlib   # optional, for PNG export
```

Ollama setup (optional):

```bash
# Install from https://ollama.ai, then pull a model
ollama pull mistral
```

---

## Input Data

Both files must be placed under `./input/`.

### `lease_data.csv` — Columns

| Column | Type | Description |
|---|---|---|
| `lease_id` | string | Unique lease identifier |
| `property_id` | string | Foreign key linking to a property record |
| `property_usage_type` | string | Usage type of the leased space (e.g. Office, Retail) |
| `total_leased_space` | numeric | Total area leased (sq ft) |
| `asking_rent_monthly` | numeric | Asking monthly rent |
| `base_rent_monthly` | numeric | Base monthly rent |
| `start_date` | `YYYY-MM-DD HH:MM:SS.f` | Lease start date |
| `end_date` | `YYYY-MM-DD HH:MM:SS.f` | Lease end date |
| `signed_date` | `YYYY-MM-DD HH:MM:SS.f` | Date the lease was signed |
| `lease_status` | string | Status label (e.g. Active, Expired) |
| `lease_type` | string | Lease activity type (e.g. New Lease, Expansion, Renewal) |

### `property_data.csv` — Columns

| Column | Type | Description |
|---|---|---|
| `property_id` | string | Unique property identifier |
| `property_usage_type` | string | Usage type label (e.g. Office, Retail, Industrial) |
| `property_class` | string | Class type label (e.g. Class A, Class B) |
| `address_line` | string | Street address |
| `city` | string | City name |
| `state` | string | State code (e.g. CA, MA) |
| `postal_code` | string | Postal / ZIP code |
| `market_name` | string | Market description (e.g. Los Angeles, Massachusetts) |
| `net_rentable_area` | numeric | Net rentable area (sq ft) |
| `total_gross_area` | numeric | Total gross area (sq ft) |
| `last_sale_price` | numeric | Last recorded sale price |

---

## Usage

### Basic run — build graph and export all files

```bash
cd python/knowledge_graph
python knowledge_graph.py
```

### Ask a single natural-language question

```bash
python knowledge_graph.py --question "Which leases are in the Los Angeles market?"
```

### Start an interactive chat session

```bash
python knowledge_graph.py --chat
```

### Use a different Ollama model

```bash
python knowledge_graph.py --model llama3 --question "Show properties with gross area > 50000"
```

### Combine question and chat flags

```bash
python knowledge_graph.py --question "List Office leases" --chat
```

> If Ollama is not running, all query and chat operations automatically fall back to deterministic lexical search — no configuration required.

---

## Module Reference

### `knowledge_graph.py`

CLI entrypoint and pipeline orchestrator. Parses arguments, loads CSV data, builds the graph, triggers optional enrichment and exports, and dispatches question/chat handling.

| Function | Description |
|---|---|
| `parse_args()` | Declares and parses `--question`, `--chat`, and `--model` CLI flags |
| `handle_cli_question(graph, question, ollama)` | Answers a single question via Ollama; falls back to lexical search if unavailable |
| `handle_chatbot_session(graph, ollama)` | Runs an interactive REPL loop, maintaining multi-turn conversation history |
| `main()` | Full pipeline: parse → build → enrich → export → query/chat |

---

### `graph_builder.py`

Builds the `networkx.MultiDiGraph` from parsed CSV rows and optionally adds LLM-generated insight nodes.

| Function | Description |
|---|---|
| `build_lease_property_kg(lease_rows, property_rows)` | Creates all nodes and edges from lease and property records |
| `enrich_with_ollama_insights(graph, ollama)` | Prompts Ollama for up to 3 business insights; adds them as `Insight` nodes connected to an `InsightCollection` hub |
| `print_graph_snapshot(graph)` | Prints node/edge counts plus sample nodes and edges to stdout |

---

### `query_engine.py`

All querying logic: text corpus construction, lexical scoring, deterministic graph traversal, LLM-assisted QA, and multi-turn chat.

| Function | Description |
|---|---|
| `graph_node_to_text(node_id, attrs)` | Serialises a node into a pipe-delimited text string |
| `graph_edge_to_text(source, target, attrs)` | Serialises an edge into a pipe-delimited text string |
| `build_graph_text_corpus(graph)` | Converts the entire graph into a list of searchable text documents |
| `search_graph_by_text(graph, query, top_k)` | Token-overlap lexical search; returns top-k scored node/edge documents |
| `format_search_results(results)` | Formats scored results as labelled text lines for LLM prompts |
| `get_market_lease_matches(graph, query)` | Deterministic traversal: `Market ← Property ← Lease` filtered by market name tokens |
| `format_market_lease_matches(matches)` | Formats market traversal results as a prompt-ready text block |
| `extract_property_gross_area_filter(query)` | Parses a numeric `total_gross_area` filter from a natural-language query |
| `_compare_number(value, op, threshold)` | Evaluates a numeric comparison using an operator string |
| `get_property_area_lease_matches(graph, query)` | Finds properties satisfying a gross-area filter and returns their linked leases |
| `format_property_area_lease_matches(matches)` | Formats property-area results as a prompt-ready text block |
| `build_deterministic_query_matches(graph, query)` | Runs all deterministic matchers and returns a combined result dict |
| `get_graph_context_for_query(graph, query, top_k)` | Merges all match types into a single context string for LLM prompting |
| `answer_graph_question(graph, question, ollama)` | Passes graph context to Ollama and returns a constrained 3–5 sentence answer |
| `chat_with_graph(graph, question, ollama, history)` | Multi-turn chat — injects system prompt, history, and graph context into each request |
| `answer_graph_question_without_llm(graph, question)` | Fallback answering using only graph traversal and lexical search |
| `generate_graph_nlp_report(graph, ollama)` | Runs a fixed set of sample questions and returns Q&A pairs for the NLP report |

---

### `exporters.py`

Handles all output file generation.

| Function | Description |
|---|---|
| `export_graph(graph, json_path, graphml_path)` | Writes a flat JSON document and a GraphML file (None-valued attributes stripped) |
| `export_graph_image(graph, image_path)` | Renders a spring-layout PNG via matplotlib; nodes are color-coded by type |
| `export_nlp_results(graph, output_path, ollama)` | Runs sample queries and writes search results plus the Q&A report to JSON |

**Node color coding in PNG**

| Node type | Color |
|---|---|
| `Lease` | Blue `#4C78A8` |
| `Property` | Orange `#F58518` |
| `UsageType` | Green `#54A24B` |
| `City` | Red `#E45756` |
| `State` | Teal `#72B7B2` |
| `Market` | Purple `#B279A2` |
| `Insight` | Pink `#FF9DA6` |
| `InsightCollection` | Brown `#9D755D` |

---

### `ollama_client.py`

Thin HTTP client for the Ollama REST API built entirely on the Python standard library (`urllib`).

**`OllamaClient` dataclass**

| Attribute | Default | Description |
|---|---|---|
| `base_url` | `http://localhost:11434` | Ollama server base URL |
| `model` | `mistral` | Model name passed to all API requests |

| Method | Signature | Description |
|---|---|---|
| `generate` | `(prompt: str) → str` | Sends a single-turn prompt; returns the response text |
| `chat` | `(messages: List[Dict]) → str` | Sends a multi-turn message list; returns the assistant reply |
| `is_available` | `() → bool` | Returns `True` if the server responds to `/api/tags` within 5 s |

---

### `data_utils.py`

Parsing and normalization helpers shared across the project.

| Function | Description |
|---|---|
| `parse_csv_rows(csv_input, read_from_file)` | Returns a list of row dicts from inline CSV text or a file path |
| `to_float(value)` | Converts a string to `float`; returns `None` for blank or missing values |
| `to_datetime(value)` | Parses `YYYY-MM-DD HH:MM:SS.f` strings and returns ISO 8601 format |
| `normalize_text(text)` | Lowercases and collapses all whitespace runs to a single space |
| `tokenize_text(text)` | Returns a list of alphanumeric tokens extracted from normalized text |

---

### `deps.py`

| Function | Description |
|---|---|
| `require_networkx()` | Lazily imports and returns the `networkx` module; raises `ModuleNotFoundError` with an install hint if missing |

---

## Output Files

| File | Format | Description |
|---|---|---|
| `lease_property_kg.json` | JSON | Flat document with serialised node and edge lists |
| `lease_property_kg.graphml` | GraphML (XML) | Standard portable graph format; open in Gephi, Cytoscape, or yEd |
| `lease_property_kg.png` | PNG | Spring-layout graph visualization with color-coded node types |
| `lease_property_graph_nlp.json` | JSON | Sample query results and Q&A report (written only when Ollama is available) |

---

## Query Examples

The query engine supports natural-language patterns for deterministic lookups:

```
"Get leases of Boston market properties"
"Show all leases in the Chicago market"
"Properties with total gross area greater than 100000"
"Leases for properties with gross area < 50000"
"Which leases are connected to Tucson properties?"
"What usage types exist in the graph?"
"Show industrial properties with last sale price above 1000000"
```

### Supported numeric comparison keywords

| Natural language | Operator |
|---|---|
| `greater than`, `more than`, `above`, `over` | `>` |
| `less than`, `below`, `under` | `<` |
| `equal to`, `equals` | `=` |
| `>=`, `<=`, `>`, `<`, `=` | symbolic pass-through |
# knowledge_graph
