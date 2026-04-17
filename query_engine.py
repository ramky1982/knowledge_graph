from __future__ import annotations

from collections import Counter
import re
from typing import Any, Dict, List, Optional

from data_utils import normalize_text, tokenize_text
from ollama_client import OllamaClient


def graph_node_to_text(node_id: str, attrs: Dict[str, Any]) -> str:
    """Serialize a graph node into a flat pipe-delimited text string for lexical search."""
    parts = [node_id]
    for key, value in attrs.items():
        if value is None:
            continue
        parts.append(f"{key} {value}")
    return " | ".join(parts)


def graph_edge_to_text(source: str, target: str, attrs: Dict[str, Any]) -> str:
    """Serialize a graph edge into a flat pipe-delimited text string for lexical search."""
    relation = attrs.get("relation", "RELATED_TO")
    parts = [f"{source} {relation} {target}"]
    for key, value in attrs.items():
        if value is None:
            continue
        parts.append(f"{key} {value}")
    return " | ".join(parts)


def build_graph_text_corpus(graph: Any) -> List[Dict[str, Any]]:
    """Convert every node and edge in the graph into a searchable text document.

    Each document is a dict with keys: kind, id, text, attrs, and (for edges)
    source and target.
    """
    corpus: List[Dict[str, Any]] = []

    for node_id, attrs in graph.nodes(data=True):
        corpus.append(
            {
                "kind": "node",
                "id": node_id,
                "text": graph_node_to_text(node_id, dict(attrs)),
                "attrs": dict(attrs),
            }
        )

    for source, target, attrs in graph.edges(data=True):
        corpus.append(
            {
                "kind": "edge",
                "id": f"{source}->{target}",
                "source": source,
                "target": target,
                "text": graph_edge_to_text(source, target, dict(attrs)),
                "attrs": dict(attrs),
            }
        )

    return corpus


def search_graph_by_text(graph: Any, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
    """Score graph nodes and edges against a query using token-overlap and return the top-k matches.

    Scoring counts shared tokens between the query and each text document. Results
    are sorted by score descending, with node matches ranked above edge matches on ties.
    """
    query_tokens = tokenize_text(query)
    if not query_tokens:
        return []

    query_counter = Counter(query_tokens)
    results: List[Dict[str, Any]] = []

    for entry in build_graph_text_corpus(graph):
        entry_tokens = tokenize_text(entry["text"])
        entry_counter = Counter(entry_tokens)
        score = sum(min(query_counter[token], entry_counter[token]) for token in query_counter)
        if score <= 0:
            continue

        results.append(
            {
                "score": score,
                **entry,
            }
        )

    results.sort(key=lambda item: (item["score"], item["kind"] == "node"), reverse=True)
    return results[:top_k]


def format_search_results(results: List[Dict[str, Any]]) -> str:
    """Format lexical search result dicts as labelled text lines suitable for LLM prompts."""
    lines: List[str] = []
    for result in results:
        if result["kind"] == "node":
            lines.append(f"NODE {result['id']}: {result['text']}")
        else:
            lines.append(f"EDGE {result['id']}: {result['text']}")
    return "\n".join(lines)


def get_market_lease_matches(graph: Any, query: str) -> List[Dict[str, Any]]:
    """Find leases for a market by traversing the graph deterministically.

    Traversal path: Market <-IN_MARKET- Property <-HAS_LEASED_PROPERTY- Lease.
    Market names are matched against non-stop tokens extracted from the query.
    """
    query_tokens = set(tokenize_text(query))
    if not query_tokens:
        return []

    stop_tokens = {
        "get", "show", "list", "find", "lease", "leases", "of", "for", "in", "the", "a", "an", "market",
    }
    target_tokens = {tok for tok in query_tokens if tok not in stop_tokens}
    if not target_tokens:
        return []

    matches: List[Dict[str, Any]] = []

    for market_node, market_attrs in graph.nodes(data=True):
        if market_attrs.get("label") != "Market":
            continue

        market_name = str(market_attrs.get("name") or market_node)
        market_tokens = set(tokenize_text(market_name))

        if not (target_tokens & market_tokens):
            continue

        properties: List[str] = []
        for prop_node, _, edge_attrs in graph.in_edges(market_node, data=True):
            if edge_attrs.get("relation") == "IN_MARKET":
                properties.append(prop_node)

        leases: List[str] = []
        for prop_node in properties:
            for lease_node, _, edge_attrs in graph.in_edges(prop_node, data=True):
                if edge_attrs.get("relation") == "HAS_LEASED_PROPERTY":
                    leases.append(lease_node)

        if properties or leases:
            matches.append(
                {
                    "market_node": market_node,
                    "market_name": market_attrs.get("name") or market_node,
                    "properties": sorted(set(properties)),
                    "leases": sorted(set(leases)),
                }
            )

    return matches


def format_market_lease_matches(matches: List[Dict[str, Any]]) -> str:
    """Format market-to-lease traversal results as text for use in QA or chat prompts."""
    if not matches:
        return ""

    lines: List[str] = ["MARKET_LEASE_MATCHES:"]
    for match in matches:
        lines.append(
            f"Market {match['market_name']}: properties={match['properties']}, leases={match['leases']}"
        )
    return "\n".join(lines)


def extract_property_gross_area_filter(query: str) -> Optional[Dict[str, Any]]:
    """Parse a numeric filter on total_gross_area from a natural-language query.

    Supports verbal operators (greater than, less than, equal to) and symbolic
    operators (>, <, >=, <=, =). Returns a dict with keys 'field', 'op', and
    'value', or None if no filter pattern is found.
    """
    normalized = normalize_text(query)

    patterns = [
        (r"total gross area\s*(?:is\s*)?(?:greater than|more than|above|over)\s*(\d+(?:\.\d+)?)", "gt"),
        (r"gross area\s*(?:is\s*)?(?:greater than|more than|above|over)\s*(\d+(?:\.\d+)?)", "gt"),
        (r"total gross area\s*(?:is\s*)?(?:less than|below|under)\s*(\d+(?:\.\d+)?)", "lt"),
        (r"gross area\s*(?:is\s*)?(?:less than|below|under)\s*(\d+(?:\.\d+)?)", "lt"),
        (r"total gross area\s*(?:is\s*)?(?:equal to|equals?)\s*(\d+(?:\.\d+)?)", "eq"),
        (r"gross area\s*(?:is\s*)?(?:equal to|equals?)\s*(\d+(?:\.\d+)?)", "eq"),
        (r"total gross area\s*(>=|<=|>|<|=)\s*(\d+(?:\.\d+)?)", "symbol"),
        (r"gross area\s*(>=|<=|>|<|=)\s*(\d+(?:\.\d+)?)", "symbol"),
    ]

    for pattern, mode in patterns:
        match = re.search(pattern, normalized)
        if not match:
            continue

        if mode == "symbol":
            op = match.group(1)
            value = float(match.group(2))
            if op == ">":
                operator = "gt"
            elif op == "<":
                operator = "lt"
            elif op == "=":
                operator = "eq"
            elif op == ">=":
                operator = "gte"
            else:
                operator = "lte"
            return {"field": "total_gross_area", "op": operator, "value": value}

        return {"field": "total_gross_area", "op": mode, "value": float(match.group(1))}

    return None


def _compare_number(value: float, op: str, threshold: float) -> bool:
    """Evaluate a numeric comparison using an operator string (gt, lt, eq, gte, lte)."""
    if op == "gt":
        return value > threshold
    if op == "lt":
        return value < threshold
    if op == "eq":
        return value == threshold
    if op == "gte":
        return value >= threshold
    if op == "lte":
        return value <= threshold
    return False


def get_property_area_lease_matches(graph: Any, query: str) -> List[Dict[str, Any]]:
    """Find leases linked to properties that satisfy a total_gross_area filter in the query.

    Extracts a numeric filter from the query text, scans all Property nodes, and
    collects incoming HAS_LEASED_PROPERTY edges. Results are sorted by area descending.
    """
    gross_area_filter = extract_property_gross_area_filter(query)
    if not gross_area_filter:
        return []

    matches: List[Dict[str, Any]] = []
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("label") != "Property":
            continue

        gross_area = attrs.get("total_gross_area")
        if not isinstance(gross_area, (int, float)):
            continue

        if not _compare_number(float(gross_area), gross_area_filter["op"], float(gross_area_filter["value"])):
            continue

        leases: List[str] = []
        for lease_node, _, edge_attrs in graph.in_edges(node_id, data=True):
            if edge_attrs.get("relation") == "HAS_LEASED_PROPERTY":
                leases.append(lease_node)

        matches.append(
            {
                "property": node_id,
                "total_gross_area": float(gross_area),
                "leases": sorted(set(leases)),
                "filter": gross_area_filter,
            }
        )

    matches.sort(key=lambda item: item["total_gross_area"], reverse=True)
    return matches


def format_property_area_lease_matches(matches: List[Dict[str, Any]]) -> str:
    """Format property gross-area filter results as text for use in QA or chat prompts."""
    if not matches:
        return ""

    filter_info = matches[0]["filter"]
    lines: List[str] = [
        f"PROPERTY_GROSS_AREA_MATCHES: total_gross_area {filter_info['op']} {filter_info['value']}"
    ]
    for match in matches:
        lines.append(
            f"Property {match['property']}: total_gross_area={match['total_gross_area']}, leases={match['leases']}"
        )
    return "\n".join(lines)


def build_deterministic_query_matches(graph: Any, query: str) -> Dict[str, Any]:
    """Collect all deterministic match results for a query into a single dict.

    Returns market_matches and property_area_matches for downstream formatting
    and export.
    """
    return {
        "market_matches": get_market_lease_matches(graph, query),
        "property_area_matches": get_property_area_lease_matches(graph, query),
    }


def get_graph_context_for_query(graph: Any, query: str, top_k: int = 8) -> str:
    """Build a combined graph context string for a natural-language query.

    Merges deterministic market and property-area matches with top-k lexical
    search results into a single text block ready for LLM prompting.
    """
    results = search_graph_by_text(graph, query, top_k=top_k)
    deterministic_matches = build_deterministic_query_matches(graph, query)
    market_context = format_market_lease_matches(deterministic_matches["market_matches"])
    property_area_context = format_property_area_lease_matches(deterministic_matches["property_area_matches"])

    if not results and not market_context and not property_area_context:
        return "No relevant graph context found."

    search_context = format_search_results(results) if results else ""
    sections = [section for section in [market_context, property_area_context, search_context] if section]
    return "\n\n".join(sections)


def answer_graph_question(graph: Any, question: str, ollama: OllamaClient, top_k: int = 8) -> str:
    """Answer a natural-language question by passing graph context to Ollama.

    Retrieves relevant context from the graph and constructs a constrained prompt
    that instructs the model to answer in 3-5 sentences using only the supplied data.
    """
    context = get_graph_context_for_query(graph, question, top_k=top_k)
    if context == "No relevant graph context found.":
        return "No relevant graph context found for the question."

    prompt = (
        "You are answering a question about a lease and property knowledge graph. "
        "Use only the supplied graph context. If the answer is not supported, say so clearly.\n\n"
        f"Question: {question}\n\n"
        f"Graph Context:\n{context}\n\n"
        "Answer in 3 to 5 concise sentences."
    )
    return ollama.generate(prompt).strip()


def chat_with_graph(
    graph: Any,
    question: str,
    ollama: OllamaClient,
    history: List[Dict[str, str]],
    top_k: int = 8,
) -> str:
    """Respond to a question in multi-turn chat mode using conversation history and graph context.

    Injects a system prompt, the full conversation history, and current graph context
    into a chat request so the model maintains continuity across turns.
    """
    context = get_graph_context_for_query(graph, question, top_k=top_k)
    system_prompt = (
        "You are a graph-aware chatbot for lease and property data. "
        "Answer using only the provided graph context and the ongoing conversation. "
        "If the answer is not supported by the graph context, say that clearly."
    )
    user_prompt = (
        f"Graph Context:\n{context}\n\n"
        f"User Question: {question}\n\n"
        "Provide a concise, direct answer."
    )
    messages = [{"role": "system", "content": system_prompt}, *history, {"role": "user", "content": user_prompt}]
    return ollama.chat(messages).strip()


def answer_graph_question_without_llm(graph: Any, question: str, top_k: int = 5) -> str:
    """Answer a question using only graph traversal and lexical search, without an LLM.

    Tries property-area filter matches first, then market traversal matches, and
    finally falls back to raw text search results. Used when Ollama is unavailable.
    """
    property_area_matches = get_property_area_lease_matches(graph, question)
    if property_area_matches:
        filter_info = property_area_matches[0]["filter"]
        all_leases = sorted({lease for item in property_area_matches for lease in item["leases"]})
        if all_leases:
            return (
                f"Leases for properties where total_gross_area {filter_info['op']} {filter_info['value']}: "
                f"{', '.join(all_leases)}"
            )
        return (
            f"Properties matched total_gross_area {filter_info['op']} {filter_info['value']}, "
            "but no linked leases were found."
        )

    market_matches = get_market_lease_matches(graph, question)
    if market_matches:
        lines: List[str] = []
        for match in market_matches:
            leases = match["leases"]
            if leases:
                lines.append(
                    f"Market {match['market_name']} has leases: {', '.join(leases)}"
                )
            else:
                lines.append(f"Market {match['market_name']} has no linked leases in the graph.")
        return "\n".join(lines)

    results = search_graph_by_text(graph, question, top_k=top_k)
    if not results:
        return "No relevant graph context found for the question."

    lines: List[str] = []
    for result in results:
        if result["kind"] == "node":
            lines.append(f"Node match: {result['id']} | score={result['score']}")
        else:
            lines.append(
                f"Edge match: {result['source']} --[{result['attrs'].get('relation', 'RELATED_TO')}]--> {result['target']} | score={result['score']}"
            )
    return "\n".join(lines)


def generate_graph_nlp_report(graph: Any, ollama: OllamaClient) -> Dict[str, Any]:
    """Run a fixed set of sample questions against the graph and return Q&A pairs.

    Used to populate the NLP report section of the exported JSON results file.
    """
    questions = [
        "Which leases are connected to Tucson properties?",
        "What usage types exist in the graph?",
        "Which properties are in the Tucson market?",
    ]

    answers = []
    for question in questions:
        answers.append(
            {
                "question": question,
                "answer": answer_graph_question(graph, question, ollama),
            }
        )

    return {
        "questions": answers,
    }
