from __future__ import annotations

import importlib
import json
from typing import Any, Dict

from deps import require_networkx
from ollama_client import OllamaClient
from query_engine import (
    build_deterministic_query_matches,
    generate_graph_nlp_report,
    get_graph_context_for_query,
    search_graph_by_text,
)


def _sanitize_attrs(attrs: Dict[str, Any]) -> Dict[str, Any]:
    """Return attrs excluding None values (GraphML does not support NoneType)."""
    return {key: value for key, value in attrs.items() if value is not None}


def _graphml_safe_copy(graph: Any) -> Any:
    """Create a graph copy with GraphML-safe attributes only."""
    nx = require_networkx()
    safe_graph = nx.MultiDiGraph()

    safe_graph.graph.update(_sanitize_attrs(dict(graph.graph)))

    for node_id, attrs in graph.nodes(data=True):
        safe_graph.add_node(node_id, **_sanitize_attrs(dict(attrs)))

    for source, target, key, attrs in graph.edges(keys=True, data=True):
        safe_graph.add_edge(source, target, key=key, **_sanitize_attrs(dict(attrs)))

    return safe_graph


def export_graph(graph: Any, json_path: str, graphml_path: str) -> None:
    """Export the graph as a flat JSON document and a GraphML file.

    The JSON file contains serialised node and edge lists. The GraphML file
    is written with None-valued attributes stripped so it is valid XML.
    """
    payload: Dict[str, Any] = {
        "name": graph.graph.get("name"),
        "nodes": [
            {"id": node, **attrs}
            for node, attrs in graph.nodes(data=True)
        ],
        "edges": [
            {"source": src, "target": tgt, **attrs}
            for src, tgt, attrs in graph.edges(data=True)
        ],
    }

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    nx = require_networkx()
    safe_graph = _graphml_safe_copy(graph)
    nx.write_graphml(safe_graph, graphml_path)


def export_graph_image(graph: Any, image_path: str) -> None:
    """Render and save a PNG image of the graph."""
    plt = importlib.import_module("matplotlib.pyplot")
    nx = require_networkx()

    if hasattr(graph, "to_undirected"):
        layout_graph = graph.to_undirected()
    else:
        layout_graph = graph

    pos = nx.spring_layout(layout_graph, seed=42, k=0.9)

    label_color_map = {
        "Lease": "#4C78A8",
        "Property": "#F58518",
        "UsageType": "#54A24B",
        "City": "#E45756",
        "State": "#72B7B2",
        "Market": "#B279A2",
        "Insight": "#FF9DA6",
        "InsightCollection": "#9D755D",
    }

    node_colors = []
    node_labels = {}
    for node_id, attrs in graph.nodes(data=True):
        node_type = attrs.get("label", "Unknown")
        node_colors.append(label_color_map.get(node_type, "#BAB0AC"))

        if node_type in {"Lease", "Property", "UsageType"}:
            node_labels[node_id] = node_id.split(":", 1)[-1]
        else:
            node_labels[node_id] = attrs.get("name") or attrs.get("code") or node_id

    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(
        layout_graph,
        pos,
        node_color=node_colors,
        node_size=1100,
        alpha=0.9,
    )
    nx.draw_networkx_edges(layout_graph, pos, alpha=0.45, arrows=False, width=1.4)
    nx.draw_networkx_labels(layout_graph, pos, labels=node_labels, font_size=8)

    plt.title("Lease-Property Knowledge Graph", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(image_path, dpi=200, bbox_inches="tight")
    plt.close()


def export_nlp_results(graph: Any, output_path: str, ollama: OllamaClient) -> None:
    """Save NLP search examples and question-answer results for the graph."""
    queries = [
        "Get leases of Boston Market properties",
        "industrial property sale price",
        "Get all leases with total leased space greater than 20000",
        "Get all leases for properties with total gross area > 100000",
    ]

    payload: Dict[str, Any] = {
        "search_examples": [],
        "qa_report": generate_graph_nlp_report(graph, ollama),
    }

    for query in queries:
        results = search_graph_by_text(graph, query, top_k=5)
        deterministic_matches = build_deterministic_query_matches(graph, query)
        context = get_graph_context_for_query(graph, query, top_k=5)
        payload["search_examples"].append(
            {
                "query": query,
                "matches": results,
                "deterministic_matches": deterministic_matches,
                "context": context,
            }
        )

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
