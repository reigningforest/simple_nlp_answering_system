"""Shared retrieval utilities for QA services."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import unicodedata

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from src.utils import get_shared_logger, load_config
from src.rag.spacy_model import ensure_spacy_model

import spacy  # type: ignore

logger = get_shared_logger(__name__)


def _strip_possessive(value: str) -> str:
    return re.sub(r"(?:'s|’s)$", "", value.strip())


def _tokenize_name(value: str) -> List[str]:
    return [token for token in re.split(r"\W+", value.lower()) if token]


def _normalize_question(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    replacements = {
        "\u2019": "'",
        "\u2018": "'",
        "\u2032": "'",
        "\u02bc": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\ufffd": "'",
    }
    for src, dst in replacements.items():
        normalized = normalized.replace(src, dst)
    return normalized


def _strip_question_prefix(value: str) -> str:
    """Remove leading punctuation/markers that confuse NER."""
    stripped = re.sub(r"^[\s\?\!\.,:;\-\u2014\u2013\"'`]+", "", value)
    return stripped if stripped else value


@dataclass
class RetrievalResult:
    question: str
    context: str
    matches: List[Any]
    snippets: List[Dict[str, Any]]
    target_name: Optional[str]
    metadata_filter: Optional[Dict[str, Any]]
    top_k: int


class RetrievalEngine:
    def __init__(self, config_override: Optional[str] = None) -> None:
        if config_override:
            config_path = Path(config_override)
            self.config = load_config(dirname=str(config_path.parent), filename=config_path.name)
        else:
            self.config = load_config(dirname="config", filename="config.yaml")

        self.pc_index_name: str = self.config["pc_index"]
        self.top_k: int = int(self.config.get("qa_top_k", 5))

        load_dotenv()

        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise EnvironmentError("PINECONE_API_KEY environment variable is required.")

        logger.info("Initialising Pinecone client and index '%s'", self.pc_index_name)
        self.pinecone = Pinecone(api_key=api_key)
        self.index = self.pinecone.Index(self.pc_index_name)

        embedder_name = self.config.get("fast_embed_name") or "BAAI/bge-small-en-v1.5"
        logger.info("Loading sentence transformer model '%s' for query embedding", embedder_name)
        self.embedder = SentenceTransformer(embedder_name)

        ner_model_name = self.config.get("ner_model", "en_core_web_lg")
        ner_model_version = str(self.config.get("ner_model_version", "3.7.1"))
        ner_storage_dir = self.config.get("ner_model_storage_dir")
        ner_model_url = self.config.get("ner_model_url")

        logger.info(
            "Ensuring spaCy model '%s' (version %s) is available",
            ner_model_name,
            ner_model_version,
        )
        model_path = ensure_spacy_model(
            model_name=ner_model_name,
            version=ner_model_version,
            storage_dir=ner_storage_dir,
            download_url=ner_model_url,
        )
        # spaCy models have a nested structure: model_path/model_name/model_name-version
        inner_model = model_path / ner_model_name / f"{ner_model_name}-{ner_model_version}"
        if inner_model.exists():
            self.nlp = spacy.load(str(inner_model))  # type: ignore[arg-type]
        else:
            self.nlp = spacy.load(str(model_path))  # type: ignore[arg-type]

    def parse_question(self, question: str) -> Tuple[str, Optional[str]]:
        normalized = _normalize_question(question).strip()
        if not normalized:
            raise ValueError("Question must not be empty.")
        ner_ready = _strip_question_prefix(normalized)
        target_name = self._extract_target_name(ner_ready)
        return normalized, target_name

    def retrieve(
        self, question: str, target_name_override: Optional[str] = None
    ) -> RetrievalResult:
        question, extracted_name = self.parse_question(question)

        logger.debug("Received question: %s", question)
        target_name = target_name_override or extracted_name
        query_vector = self.embedder.encode(question).tolist()

        filter_clause = self._build_metadata_filter(target_name)
        logger.debug("Query filter: %s", filter_clause)

        query_kwargs: Dict[str, Any] = {
            "vector": query_vector,
            "top_k": max(self.top_k, 20),
            "include_metadata": True,
        }
        if filter_clause:
            query_kwargs["filter"] = filter_clause

        logger.info("Querying Pinecone index")
        results = self.index.query(**query_kwargs)
        matches = self._extract_matches(results)

        context, snippets = self._build_context(matches)
        logger.debug("Assembled context length: %d characters", len(context))

        return RetrievalResult(
            question=question,
            context=context,
            matches=matches,
            snippets=snippets,
            target_name=target_name,
            metadata_filter=filter_clause,
            top_k=self.top_k,
        )

    def _extract_target_name(self, question: str) -> Optional[str]:
        doc = self.nlp(question)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        logger.debug("spaCy entities detected: %s", entities or "<none>")
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                candidate = _strip_possessive(ent.text)
                return candidate.strip() or None
        logger.debug("No PERSON entity identified in question: %s", question)
        return None

    def _build_metadata_filter(self, target_name: Optional[str]) -> Optional[Dict[str, Any]]:
        if not target_name:
            return None

        normalized = _strip_possessive(target_name).lower().strip()
        tokens = _tokenize_name(normalized)

        clauses: List[Dict[str, Any]] = []
        if normalized:
            clauses.append({"user_name_normalized": {"$eq": normalized}})
        clauses.extend({"user_name_tokens": {"$in": [token]}} for token in tokens)

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def _build_context(self, matches: Iterable[Any]) -> Tuple[str, List[Dict[str, Any]]]:
        ordered: List[Tuple[Optional[datetime], int, str]] = []
        snippets: List[Dict[str, Any]] = []
        for idx, match in enumerate(matches):
            if isinstance(match, dict):
                metadata = match.get("metadata")
            else:
                metadata = getattr(match, "metadata", None)
            if not isinstance(metadata, dict):
                continue

            text = metadata.get("text") or metadata.get("message")
            if not isinstance(text, str) or not text.strip():
                continue

            raw_timestamp = metadata.get("timestamp")
            user_name = metadata.get("user_name")
            display_user = user_name.strip() if isinstance(user_name, str) else "Unknown member"
            parsed_timestamp = self._parse_timestamp(raw_timestamp)
            timestamp_label = f"[{raw_timestamp}] " if raw_timestamp else ""
            ordered.append((parsed_timestamp, idx, f"{timestamp_label}{display_user}: {text.strip()}"))
            snippets.append(
                {
                    "timestamp": raw_timestamp,
                    "parsed_timestamp": parsed_timestamp,
                    "user_name": user_name,
                    "text": text.strip(),
                }
            )

        ordered.sort(key=lambda item: (item[0] or datetime.max, item[1]))

        ordered_strings = [entry for _, _, entry in ordered[: self.top_k]]
        snippets = snippets[: self.top_k]
        return "\n\n".join(ordered_strings), snippets

    @staticmethod
    def _parse_timestamp(value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(float(value), tz=timezone.utc).replace(tzinfo=None)
            except (OverflowError, OSError, ValueError):
                return None
        if not isinstance(value, str):
            return None

        candidate = value.strip()
        if not candidate:
            return None

        candidate_iso = candidate.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(candidate_iso)
            if parsed.tzinfo is not None:
                return parsed.astimezone(timezone.utc).replace(tzinfo=None)
            return parsed
        except ValueError:
            pass

        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y %H:%M", "%m/%d/%Y"):
            try:
                return datetime.strptime(candidate, fmt)
            except ValueError:
                continue
        return None

    @staticmethod
    def _extract_matches(results: Any) -> List[Any]:
        if results is None:
            return []
        if isinstance(results, dict):
            return list(results.get("matches", []))
        matches = getattr(results, "matches", None)
        if matches is None:
            return []
        if isinstance(matches, list):
            return matches
        return list(matches)