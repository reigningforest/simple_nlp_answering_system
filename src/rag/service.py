"""Retrieval-augmented question answering service logic."""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from difflib import get_close_matches
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import requests

from src.rag.retriever import RetrievalEngine, RetrievalResult
from src.utils import get_shared_logger

logger = get_shared_logger(__name__)


BASE_DIR = Path(__file__).parent.parent.parent
KNOWN_NAMES_PATH = BASE_DIR / "config" / "known_names.json"
@dataclass
class MemberResolution:
    display_name: Optional[str]
    normalized_filter: Optional[str]
    error: Optional[str]


def _normalize_member_name(value: str) -> str:
    cleaned = value.strip()
    cleaned = re.sub(r"(?:'s|’s)$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.lower()


def _load_known_names() -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    if not KNOWN_NAMES_PATH.exists():
        logger.warning(
            "Known names file missing at %s. Run run_one_time/get_known_names.py to generate it.",
            KNOWN_NAMES_PATH,
        )
        return {}, {}

    try:
        raw_entries = json.loads(KNOWN_NAMES_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse known names file: %s", exc)
        return {}, {}

    normalized_map: Dict[str, str] = {}
    first_name_index: Dict[str, List[str]] = defaultdict(list)
    for entry in raw_entries:
        raw_name = entry.get("raw")
        normalized_name = entry.get("normalized")
        if not isinstance(raw_name, str) or not raw_name.strip():
            continue
        normalized = (
            normalized_name
            if isinstance(normalized_name, str) and normalized_name.strip()
            else _normalize_member_name(raw_name)
        )
        normalized_map.setdefault(normalized, raw_name.strip())
        first_token = normalized.split()[0] if normalized else ""
        if first_token:
            if raw_name.strip() not in first_name_index[first_token]:
                first_name_index[first_token].append(raw_name.strip())
    return normalized_map, first_name_index


def _load_text_file(path: Path, fallback: str) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.warning("Prompt file not found at %s, using fallback text", path)
        return fallback


def _load_system_prompt() -> str:
    """Load system prompt from file, falling back to default if not found."""
    prompt_path = BASE_DIR / "prompts" / "system_prompt.txt"
    return _load_text_file(
        prompt_path,
        (
            "You are a helpful assistant that answers questions about community messages. "
            "Use the provided messaage to answer the question."
        ),
    )


def _load_user_template() -> str:
    template_path = BASE_DIR / "prompts" / "user_prompt.txt"
    fallback = (
        "Message summary:\n"
        "- Member referenced in question: {member_name}\n"
        "- Latest recorded activity: {latest_activity}\n"
        "- Snippets retrieved: {snippet_count}\n\n"
        "Messagge (oldest → newest, all from this member):\n{context}\n\n"
        "Question: {question}\n\n"
        "Respond exactly in this format:\n"
        "Reasoning: <one sentence citing the most relevant snippet>\n"
        "Answer: <final concise answer>"
    )
    return _load_text_file(template_path, fallback)


class QAService:
    """Encapsulates embedding, retrieval, and generation for question answering."""

    def __init__(self) -> None:
        config_override = os.environ.get("QA_CONFIG_PATH")
        self.retriever = RetrievalEngine(config_override=config_override)
        self.config = self.retriever.config
        self.top_k = self.retriever.top_k

        self.groq_api_key = os.environ.get("GROQ_API_KEY")
        self.groq_model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

        if not self.groq_api_key:
            raise EnvironmentError(
                "GROQ_API_KEY environment variable is required for LLM generation."
            )

        self.system_prompt = _load_system_prompt()
        self.message_template = _load_user_template()
        (
            self.known_names_map,
            self.first_name_index,
        ) = _load_known_names()
        logger.info("Using Groq model '%s' for generation", self.groq_model)

    def get_answer(self, question: str) -> str:
        normalized_question, detected_name = self.retriever.parse_question(question)
        resolution = self._resolve_member_name(detected_name)
        if resolution.error:
            return resolution.error

        retrieval = self.retriever.retrieve(
            normalized_question,
            target_name_override=resolution.normalized_filter,
        )
        if resolution.display_name:
            retrieval.target_name = resolution.display_name
        return self._call_groq(
            question=retrieval.question,
            context=retrieval.context,
            retrieval=retrieval,
        )

    def _call_groq(self, question: str, context: str, retrieval: RetrievalResult) -> str:
        """Call Groq API using OpenAI-compatible chat completions format."""
        context_section = context if context else "No relevant context was retrieved."
        summary_vars = self._build_context_summary(retrieval)
        
        user_content = self.message_template.format(
            member_name=summary_vars["member_name"],
            latest_activity=summary_vars["latest_activity"],
            snippet_count=summary_vars["snippet_count"],
            context=context_section,
            question=question,
        )

        logger.debug("Final user prompt sent to LLM:\n%s", user_content)

        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": user_content,
            }
        ]

        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.groq_model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 256,
            "top_p": 0.9,
        }

        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=30,
            )

            if response.status_code >= 400:
                raise RuntimeError(
                    f"Groq API request failed: {response.status_code} - {response.text}"
                )

            data = response.json()
            
            if "choices" not in data or not data["choices"]:
                raise RuntimeError("Groq API returned no choices.")
            
            message_content = data["choices"][0].get("message", {}).get("content")
            
            if not message_content:
                raise RuntimeError("Groq API returned no content.")
            
            final_answer = message_content.strip()
            processed_answer = self._extract_final_answer(final_answer)
            logger.info(
                "LLM raw answer for question '%s': %s",
                question,
                final_answer,
            )
            logger.info(
                "LLM processed answer for question '%s': %s",
                question,
                processed_answer,
            )
            return processed_answer
            
        except requests.exceptions.RequestException as exc:
            logger.error("Groq API request failed: %s", exc)
            raise RuntimeError(f"Failed to communicate with Groq API: {exc}") from exc

    @staticmethod
    def _build_context_summary(retrieval: RetrievalResult) -> Dict[str, str]:
        member_name = retrieval.target_name or "Name not explicitly mentioned; assume the member in the question."
        snippet_count = len(retrieval.snippets)
        latest_activity = "No activity found."
        if retrieval.snippets:
            latest = retrieval.snippets[-1]
            ts = latest.get("timestamp") or "timestamp not provided"
            text = latest.get("text", "").strip()
            preview = text if len(text) <= 160 else text[:157] + "..."
            latest_activity = f"{ts} - {preview}" if text else ts

        return {
            "member_name": member_name,
            "latest_activity": latest_activity,
            "snippet_count": str(snippet_count),
        }

    def _resolve_member_name(self, detected_name: Optional[str]) -> MemberResolution:
        if not detected_name:
            return MemberResolution(display_name=None, normalized_filter=None, error=None)

        normalized = _normalize_member_name(detected_name)
        if not self.known_names_map:
            return MemberResolution(detected_name, normalized, None)

        if normalized in self.known_names_map:
            return MemberResolution(self.known_names_map[normalized], normalized, None)

        tokens = normalized.split()
        if tokens:
            first_token = tokens[0]
            matches = self.first_name_index.get(first_token, [])
            if len(matches) == 1:
                match = matches[0]
                return MemberResolution(match, _normalize_member_name(match), None)

        suggestions = self._suggest_names(normalized)
        message = self._format_invalid_name_message(suggestions)
        return MemberResolution(display_name=None, normalized_filter=None, error=message)

    @staticmethod
    def _extract_final_answer(raw_response: str) -> str:
        if not raw_response:
            return raw_response

        lowered = raw_response.lower()
        marker = "answer:"
        idx = lowered.rfind(marker)
        if idx != -1:
            extracted = raw_response[idx + len(marker) :].strip()
            if extracted:
                return extracted
        return raw_response.strip()

    def _suggest_names(self, normalized: str, limit: int = 5) -> List[str]:
        if not normalized or not self.known_names_map:
            return []
        candidates = list(self.known_names_map.keys())
        matches = get_close_matches(normalized, candidates, n=limit, cutoff=0.5)
        return [self.known_names_map[match] for match in matches]

    @staticmethod
    def _format_invalid_name_message(suggestions: List[str]) -> str:
        if suggestions:
            return "Enter a valid name. Closest matches: " + ", ".join(suggestions)
        return "Enter a valid name. No close matches found."


_qa_service: Optional[QAService] = None


def _get_service() -> QAService:
    global _qa_service
    if _qa_service is None:
        _qa_service = QAService()
    return _qa_service


def get_answer(question: str) -> str:
    """Return an answer for the supplied question."""
    service = _get_service()
    return service.get_answer(question)
