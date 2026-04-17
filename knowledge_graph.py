#!/usr/bin/env python3
"""Build a knowledge graph from Lease and Property data."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional
from urllib import error as urlerror

from data_utils import parse_csv_rows
from deps import require_networkx
from exporters import export_graph, export_graph_image, export_nlp_results
from graph_builder import build_lease_property_kg, enrich_with_ollama_insights, print_graph_snapshot
from ollama_client import OllamaClient
from query_engine import (
    answer_graph_question,
    answer_graph_question_without_llm,
    chat_with_graph,
)


# Optional: set these to CSV file paths to read data from files instead of inline text.
LEASE_DATA_FILE: Optional[str] = "./input/lease_data.csv"
PROPERTY_DATA_FILE: Optional[str] = "./input/property_data.csv"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for graph generation and NLP querying."""
    parser = argparse.ArgumentParser(
        description="Build a lease-property knowledge graph and query it with natural language.",
    )
    parser.add_argument(
        "--question",
        dest="question",
        help="Ask a natural-language question against the graph.",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Start an interactive chatbot session against the graph.",
    )
    parser.add_argument(
        "--model",
        default="mistral",
        help="Ollama model to use for enrichment, QA, and chat. Default: mistral.",
    )
    return parser.parse_args()


def handle_cli_question(graph: Any, question: str, ollama: Optional[OllamaClient]) -> None:
    """Handle a user-supplied CLI question against the graph."""
    print("\n=== Graph Question ===")
    print(f"Question: {question}")

    if ollama is not None:
        try:
            answer = answer_graph_question(graph, question, ollama)
            print("Answer:")
            print(answer)
            return
        except (urlerror.URLError, TimeoutError, ValueError, json.JSONDecodeError):
            pass

    print("Answer (fallback lexical search):")
    print(answer_graph_question_without_llm(graph, question))


def handle_chatbot_session(graph: Any, ollama: Optional[OllamaClient]) -> None:
    """Run an interactive chatbot loop over the graph."""
    print("\n=== Graph Chatbot ===")
    print("Type a question about the lease-property graph. Type 'exit' or 'quit' to stop.")

    history: List[Dict[str, str]] = []
    ollama_available = ollama is not None

    while True:
        try:
            question = input("You: ").strip()
        except EOFError:
            print("\nChat ended.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Chat ended.")
            break

        if ollama_available:
            answer = chat_with_graph(graph, question, ollama, history)
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
        else:
            answer = answer_graph_question_without_llm(graph, question)

        print(f"Bot: {answer}")


def main() -> None:
    """Run the full knowledge graph pipeline.

    Loads lease and property CSV data, builds the graph, optionally enriches it
    with Ollama-generated insights, exports to JSON/GraphML/PNG, and handles
    any CLI question or interactive chat session.
    """
    args = parse_args()
    ollama_client = OllamaClient(model=args.model)

    try:
        require_networkx()
    except ModuleNotFoundError:
        print("ERROR: networkx is required for this script.")
        print("Install it with: pip install networkx")
        return

    lease_rows = parse_csv_rows(
        LEASE_DATA_FILE,
        read_from_file=bool(LEASE_DATA_FILE),
    )
    property_rows = parse_csv_rows(
        PROPERTY_DATA_FILE,
        read_from_file=bool(PROPERTY_DATA_FILE),
    )

    graph = build_lease_property_kg(lease_rows, property_rows)
    ollama_available = ollama_client.is_available()

    if ollama_available:
        try:
            enrich_with_ollama_insights(graph, ollama_client)
            ollama_status = "enabled"
        except (urlerror.URLError, TimeoutError, ValueError, json.JSONDecodeError):
            ollama_status = "enabled (insight enrichment failed)"
    else:
        ollama_status = "skipped (Ollama unavailable)"

    if ollama_available:
        try:
            export_nlp_results(graph, "lease_property_graph_nlp.json", ollama_client)
            nlp_status = "saved"
        except (urlerror.URLError, TimeoutError, ValueError, json.JSONDecodeError):
            nlp_status = "skipped (NLP export failed)"
    else:
        nlp_status = "skipped (Ollama unavailable)"

    print_graph_snapshot(graph)
    export_graph(graph, "lease_property_kg.json", "lease_property_kg.graphml")

    try:
        export_graph_image(graph, "lease_property_kg.png")
        image_status = "saved"
    except ModuleNotFoundError:
        image_status = "skipped (matplotlib unavailable)"

    if args.question:
        handle_cli_question(graph, args.question, ollama_client if ollama_available else None)

    if args.chat:
        handle_chatbot_session(graph, ollama_client if ollama_available else None)

    print("\n=== Build Complete ===")
    print(f"Ollama enrichment: {ollama_status}")
    print("Exported: lease_property_kg.json")
    print("Exported: lease_property_kg.graphml")
    print(f"Graph NLP report: {nlp_status}")
    print(f"Graph image: {image_status}")


if __name__ == "__main__":
    main()
