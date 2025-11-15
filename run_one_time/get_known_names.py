"""Extract unique member names from all_messages.json for quick validation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MESSAGES_PATH = DATA_DIR / "all_messages.json"
OUTPUT_PATH = "config/known_names.json"


def _normalize_name(value: str) -> str:
    cleaned = value.strip()
    cleaned = re.sub(r"(?:'s|’s)$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.lower()


def build_known_names() -> Dict[str, str]:
    if not MESSAGES_PATH.exists():
        raise FileNotFoundError(
            f"Messages file not found at {MESSAGES_PATH}. Run get_messages.py first."
        )

    with MESSAGES_PATH.open("r", encoding="utf-8") as handle:
        messages = json.load(handle)

    mapping: Dict[str, str] = {}
    for item in messages:
        user_name = item.get("user_name")
        if not isinstance(user_name, str) or not user_name.strip():
            continue
        normalized = _normalize_name(user_name)
        mapping.setdefault(normalized, user_name.strip())
    return mapping


def main() -> None:
    mapping = build_known_names()
    entries = [
        {"normalized": normalized, "raw": raw}
        for normalized, raw in sorted(mapping.items(), key=lambda kv: kv[1].lower())
    ]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2, ensure_ascii=False)
    print(f"Wrote {len(entries)} unique names to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
