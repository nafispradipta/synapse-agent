"""
Synapse Neural Embedding Engine.
Provides lightweight neural dense embeddings with pure-Python fallback.
Supports sentence-transformers (all-MiniLM-L6-v2) when available, otherwise
falls back gracefully to a deterministic CRC32 n-gram vectorizer without crashing.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import os
import re
import threading
import urllib.request
import urllib.error
import zlib
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# Default model for the sentence-transformers backend
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Lazy availability checks without eager heavy imports --------------------
def _check_package_available(pkg: str) -> bool:
    try:
        return importlib.util.find_spec(pkg) is not None
    except (ImportError, AttributeError, ValueError):
        return False

_ONNX_AVAILABLE: bool = _check_package_available("onnxruntime")
_ST_AVAILABLE: bool = _check_package_available("sentence_transformers")


def fallback_vectorize(text: str, dim: int = 384) -> List[float]:
    """
    Deterministic subword n-gram L2-normalized vectorizer.
    Zero external dependencies, ultra-fast fallback. Default dimension is 384
    to ensure full structural compatibility with all-MiniLM-L6-v2 embeddings.
    """
    if not text or not text.strip():
        return [0.0] * dim

    clean_text = text.lower().strip()
    words = re.findall(r"[a-z0-9_\-]+", clean_text)
    if not words:
        return [0.0] * dim

    vec = [0.0] * dim
    for w in words:
        h = zlib.crc32(w.encode("utf-8"))
        bucket = h % dim
        sign = 1.0 if ((h >> 16) & 1) else -1.0
        vec[bucket] += sign * 1.0
        if len(w) >= 3:
            for n in (3, 4):
                for i in range(len(w) - n + 1):
                    sub = w[i : i + n]
                    sh = zlib.crc32(sub.encode("utf-8"))
                    sbucket = sh % dim
                    ssign = 1.0 if ((sh >> 16) & 1) else -1.0
                    vec[sbucket] += ssign * 0.5

    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0.0:
        vec = [x / norm for x in vec]
    return vec


class SynapseEmbedder:
    """
    Lightweight Dense Neural Embedder with automatic fallback and lazy loading.

    Primary backend: sentence-transformers (all-MiniLM-L6-v2, 384-dim).
    When sentence-transformers is installed and loads successfully, embed()
    returns vectors of the model's native dimension (384 for MiniLM).

    Fallback backend: if sentence-transformers is not installed or fails to
    load, the deterministic CRC32 n-gram vectorizer is used and vectors are
    produced at the requested ``dim`` (default 384).

    Model loading is strictly lazy: heavy weights and PyTorch/transformers are
    never initialized upon instantiation, but deferred until the first .embed() call.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        dim: int = 384,
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_model: Optional[str] = None,
    ) -> None:
        self.dim = dim
        self.model_name = model_name or os.environ.get("SYNAPSE_EMBEDDING_MODEL") or DEFAULT_MODEL_NAME
        self.model_path = model_path
        self.api_base_url = (
            api_base_url
            or os.environ.get("OPENAI_EMBEDDING_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("SYNAPSE_BASE_URL")
        )
        self.api_key = (
            api_key
            or os.environ.get("OPENAI_EMBEDDING_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("SYNAPSE_API_KEY")
            or ""
        )
        self.api_model = api_model or os.environ.get("OPENAI_EMBEDDING_MODEL")
        self._session: Optional[Any] = None  # legacy ONNX session
        self._model: Optional[Any] = None    # sentence-transformers model
        self._is_neural: bool = False
        self._native_dim: Optional[int] = 384 if _ST_AVAILABLE else None
        self._loaded: bool = False
        self._load_lock = threading.Lock()

    def _embed_remote_api(self, text: str) -> Optional[List[float]]:
        """Call remote OpenAI-compatible /v1/embeddings endpoint if configured."""
        if not self.api_base_url or not self.api_model:
            return None
        url = f"{self.api_base_url.rstrip('/')}/embeddings"
        payload = json.dumps({
            "input": [text if text else " "],
            "model": self.api_model,
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                vec = data["data"][0]["embedding"]
                norm = math.sqrt(sum(x * x for x in vec))
                return [x / norm for x in vec] if norm > 0.0 else vec
        except Exception as e:
            logger.debug("Remote embedding API call failed: %s; falling back.", e)
            return None

    def _ensure_loaded(self) -> None:
        """Lazily initialize neural backend upon first demand."""
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            self._loaded = True

            # 1) Try sentence-transformers (primary backend)
            if _ST_AVAILABLE:
                try:
                    from sentence_transformers import SentenceTransformer as _STModel
                    load_target = (
                        self.model_path
                        if (self.model_path and os.path.exists(self.model_path))
                        else self.model_name
                    )
                    self._model = _STModel(load_target)
                    probe = self._model.encode(
                        ["dimension probe"],
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                    )
                    self._native_dim = int(probe.shape[1])
                    self._is_neural = True
                    logger.info(
                        "Synapse sentence-transformers embedder initialized "
                        "(model=%s, native_dim=%d)",
                        load_target,
                        self._native_dim,
                    )
                    return
                except Exception as e:
                    logger.warning(
                        "Failed to load sentence-transformers model (%s): %s "
                        "Falling back to CRC32 n-gram vectorizer.",
                        self.model_name,
                        e,
                    )
                    self._model = None
                    self._is_neural = False
                    self._native_dim = None

            # 2) Legacy ONNX backend (kept for backward compatibility)
            if _ONNX_AVAILABLE and self.model_path and os.path.exists(self.model_path):
                try:
                    import onnxruntime as ort
                    opts = ort.SessionOptions()
                    opts.intra_op_num_threads = 1
                    opts.inter_op_num_threads = 1
                    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    self._session = ort.InferenceSession(
                        self.model_path,
                        sess_options=opts,
                        providers=["CPUExecutionProvider"],
                    )
                    self._is_neural = True
                    logger.info(
                        "Synapse ONNX neural embedder initialized from %s",
                        self.model_path,
                    )
                    return
                except Exception as e:
                    logger.warning(
                        "Failed to load ONNX model (%s), using fallback vectorizer: %s",
                        self.model_path,
                        e,
                    )
                    self._session = None
                    self._is_neural = False

    @property
    def model(self) -> Any:
        self._ensure_loaded()
        return self._model

    @model.setter
    def model(self, value: Any) -> None:
        self._model = value
        if value is not None:
            self._is_neural = True

    @property
    def session(self) -> Any:
        self._ensure_loaded()
        return self._session

    @session.setter
    def session(self, value: Any) -> None:
        self._session = value
        if value is not None:
            self._is_neural = True

    @property
    def is_neural(self) -> bool:
        self._ensure_loaded()
        return self._is_neural

    @is_neural.setter
    def is_neural(self, value: bool) -> None:
        self._is_neural = value

    @property
    def native_dim(self) -> Optional[int]:
        """Return the model's native embedding dimension when neural, else None."""
        if not self._loaded and _ST_AVAILABLE:
            return 384
        return self._native_dim

    def embed(self, text: str) -> List[float]:
        """Generate embedding vector.

        Tier 1: Remote OpenAI-compatible /v1/embeddings endpoint (if configured).
        Tier 2: Local sentence-transformers model (if installed, 384-dim native).
        Tier 3: Built-in deterministic subword n-gram vectorizer (zero dependencies).
        """
        # Tier 1: Check remote embedding API if configured
        remote_vec = self._embed_remote_api(text)
        if remote_vec is not None:
            return remote_vec

        self._ensure_loaded()

        # Tier 2: sentence-transformers
        if self._is_neural and self._model is not None:
            try:
                vec = self._model.encode(
                    [text if text else " "],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )[0].tolist()
                # L2-normalize defensively (MiniLM with normalize=True is already unit).
                norm = math.sqrt(sum(x * x for x in vec))
                if norm > 0.0:
                    vec = [x / norm for x in vec]
                return vec
            except Exception as e:
                logger.warning("sentence-transformers embed failed (%s); using fallback.", e)

        # Legacy ONNX backend
        if self._is_neural and self._session is not None:
            try:
                import numpy as np
                clean = text.lower().strip()
                tokens = [ord(c) % 256 for c in clean[:128]]
                if not tokens:
                    return [0.0] * self.dim
                tokens += [0] * (128 - len(tokens))
                input_ids = np.array([tokens], dtype=np.int64)
                outputs = self._session.run(None, {"input_ids": input_ids})
                vec = outputs[0][0].tolist()[: self.dim]
                norm = math.sqrt(sum(x * x for x in vec))
                return [x / norm for x in vec] if norm > 0.0 else vec
            except Exception:
                pass

        # Fallback: deterministic CRC32 n-gram vectorizer
        return fallback_vectorize(text, dim=self.dim)
