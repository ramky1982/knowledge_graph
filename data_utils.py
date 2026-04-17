from __future__ import annotations

import csv
import io
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def parse_csv_rows(csv_input: str, read_from_file: bool = False) -> List[Dict[str, str]]:
    """Parse CSV rows from inline CSV text or a file path."""
    if read_from_file:
        csv_path = Path(csv_input)
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV file not found: {csv_input}")
        csv_text = csv_path.read_text(encoding="utf-8")
    else:
        csv_text = csv_input

    return list(csv.DictReader(io.StringIO(csv_text.strip())))


def to_float(value: Optional[str]) -> Optional[float]:
    """Safely parse numeric text into float."""
    if value is None:
        return None
    raw = value.strip()
    if raw == "":
        return None
    return float(raw)


def to_datetime(value: Optional[str]) -> Optional[str]:
    """Normalize datetime string to ISO format for graph attributes."""
    if value is None:
        return None
    raw = value.strip()
    if raw == "":
        return None
    parsed = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S.%f")
    return parsed.isoformat()


def normalize_text(text: str) -> str:
    """Normalize free text for lightweight NLP matching."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokenize_text(text: str) -> List[str]:
    """Tokenize text for simple lexical search over graph content."""
    return re.findall(r"[a-z0-9_]+", normalize_text(text))
