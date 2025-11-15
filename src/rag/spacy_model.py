"""Utility helpers for downloading and caching spaCy models at runtime."""

from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Optional

import requests

from src.utils import get_shared_logger

logger = get_shared_logger(__name__)

_BASE_URL = "https://github.com/explosion/spacy-models/releases/download"
_STORAGE_ENV = "SPACY_MODEL_DIR"


class SpaCyModelDownloadError(RuntimeError):
    """Raised when the spaCy model cannot be downloaded or extracted."""


def ensure_spacy_model(
    model_name: str,
    version: str,
    storage_dir: Optional[str] = None,
    download_url: Optional[str] = None,
) -> Path:
    """Download the requested spaCy model into persistent storage if needed.

    Args:
        model_name: spaCy model package name (e.g., "en_core_web_lg").
        version: Model version string (e.g., "3.7.1").
        storage_dir: Optional explicit directory for cached models.
        download_url: Optional explicit URL to fetch the model archive.

    Returns:
        Path to the local directory that contains the extracted model.
    """

    if not model_name:
        raise ValueError("model_name is required when ensuring spaCy model availability.")

    resolved_storage = Path(
        storage_dir or os.environ.get(_STORAGE_ENV, "./runtime_models/spacy")
    ).expanduser().resolve()
    resolved_storage.mkdir(parents=True, exist_ok=True)

    version_label = version or "latest"
    target_dir = resolved_storage / f"{model_name}-{version_label}"
    meta_file = target_dir / "meta.json"
    if meta_file.exists():
        logger.info("Using cached spaCy model at %s", target_dir)
        return target_dir

    archive_url = download_url or _build_default_url(model_name, version)
    logger.info(
        "Downloading spaCy model '%s' (%s) from %s",
        model_name,
        version or "unspecified",
        archive_url,
    )

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)
            archive_path = temp_dir / "model.tar.gz"
            _download_file(archive_url, archive_path)

            extract_root = temp_dir / "extract"
            extract_root.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(extract_root)

            model_root = _find_model_root(extract_root)
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.move(str(model_root), target_dir)
    except Exception as exc:  # noqa: BLE001 - surface full context to caller
        raise SpaCyModelDownloadError(
            f"Failed to download spaCy model '{model_name}' ({version_label}): {exc}"
        ) from exc

    logger.info("spaCy model stored at %s", target_dir)
    return target_dir


def _build_default_url(model_name: str, version: str | None) -> str:
    if not version:
        raise ValueError("Model version is required when no explicit download URL is provided.")
    return f"{_BASE_URL}/{model_name}-{version}/{model_name}-{version}.tar.gz"


def _download_file(url: str, destination: Path) -> None:
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1_048_576):
                if chunk:
                    handle.write(chunk)


def _find_model_root(extracted_root: Path) -> Path:
    candidates = [path.parent for path in extracted_root.rglob("meta.json")]
    if not candidates:
        raise RuntimeError("spaCy archive did not contain a meta.json file; cannot locate model root.")
    # The tarball usually contains a single top-level directory; take the first match.
    return candidates[0]
