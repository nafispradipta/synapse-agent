import os
import json
import time
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger("synapse.config")

# Sane defaults - purely environment-driven (no hardcoded URLs, keys, or models)
DEFAULT_PRIMARY_BASE_URL: str = ""
DEFAULT_MODEL: str = ""
DEFAULT_API_KEY: str = ""


def get_llm_base_urls() -> List[str]:
    """
    Returns prioritized list of LLM Base URLs:
    Precedence: SYNAPSE_BASE_URL (specific override) -> OPENAI_BASE_URL (standard global) -> BASE_URL -> LLM_BASE_URL.
    """
    urls: List[str] = []
    for env_var in ["SYNAPSE_BASE_URL", "OPENAI_BASE_URL", "BASE_URL", "LLM_BASE_URL"]:
        val = os.environ.get(env_var)
        if val and val.strip():
            cleaned = val.strip().rstrip("/")
            if cleaned not in urls:
                urls.append(cleaned)

    return urls


def get_llm_base_url() -> str:
    """
    General OpenAI-compatible Base URL for Synapse LLM operations.
    Returns configured environment URL, or empty string if not configured.
    """
    urls = get_llm_base_urls()
    return urls[0] if urls else ""


def get_llm_api_key(force_reload: bool = False) -> str:
    """
    General OpenAI-compatible API Key for Synapse LLM operations.
    Standard env precedence:
    1. SYNAPSE_API_KEY
    2. OPENAI_API_KEY
    3. API_KEY
    4. LLM_API_KEY
    Fallback: default placeholder key
    """
    for env_var in [
        "SYNAPSE_API_KEY",
        "OPENAI_API_KEY",
        "API_KEY",
        "LLM_API_KEY",
    ]:
        val = os.environ.get(env_var)
        if val and val.strip():
            return val.strip()

    return DEFAULT_API_KEY


def get_llm_model() -> str:
    """
    General model name for Synapse LLM operations.
    Standard env precedence:
    1. SYNAPSE_MODEL
    2. OPENAI_MODEL
    3. MODEL_NAME
    4. MODEL
    5. LLM_MODEL
    Default: empty string (let provider use active model)
    """
    for env_var in ["SYNAPSE_MODEL", "OPENAI_MODEL", "MODEL_NAME", "MODEL", "LLM_MODEL"]:
        val = os.environ.get(env_var)
        if val and val.strip():
            return val.strip()

    return DEFAULT_MODEL


# Backward compatibility alias
get_router_api_key = get_llm_api_key


def call_llm_resilient(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 2000,
    stream: bool = False,
    timeout: float = 60.0,
    max_retries: int = 3,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Optional[str]:
    """
    Resilient LLM completion caller with:
    - Automatic endpoint traversal and Bearer authentication
    - Exponential backoff retry mechanism (up to 3x) for transient network or rate errors
    """
    def _log(msg: str):
        if log_fn:
            log_fn(msg)
        else:
            logger.info(msg)

    target_model = model or get_llm_model()
    payload: Dict[str, Any] = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if target_model:
        payload["model"] = target_model
    encoded_payload = json.dumps(payload).encode("utf-8")
    base_urls = get_llm_base_urls()
    if not base_urls:
        _log("[DEBUG] No LLM base URL configured in environment (SYNAPSE_BASE_URL or OPENAI_BASE_URL). Skipping LLM completion.")
        return None

    for base_url in base_urls:
        endpoint = f"{base_url.rstrip('/')}/chat/completions"
        for attempt in range(1, max_retries + 1):
            api_key = get_llm_api_key(force_reload=(attempt > 1))
            headers = {"Content-Type": "application/json"}
            if api_key and api_key != "default_key":
                headers["Authorization"] = f"Bearer {api_key}"

            req = urllib.request.Request(endpoint, data=encoded_payload, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw_body = resp.read().decode("utf-8")
                    data = json.loads(raw_body)
                    choices = data.get("choices", [])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "")
                        return content
                    return ""
            except urllib.error.HTTPError as http_err:
                status_code = http_err.code
                if status_code in (429, 500, 502, 503, 504):
                    _log(f"[WARN] HTTP {status_code} from {endpoint} (attempt {attempt}/{max_retries}). Retrying...")
                else:
                    _log(f"[ERROR] HTTP {status_code} from {endpoint}: {http_err.reason}")
                    break

                if attempt < max_retries:
                    backoff = min(4.0, 1.0 * (2 ** (attempt - 1)))
                    time.sleep(backoff)
            except urllib.error.URLError as url_err:
                _log(f"[WARN] Connection error to {endpoint} (attempt {attempt}/{max_retries}): {url_err.reason}")
                if attempt < max_retries:
                    backoff = min(4.0, 1.0 * (2 ** (attempt - 1)))
                    time.sleep(backoff)
                else:
                    _log(f"[INFO] Exhausted retries on {endpoint}. Trying next fallback URL if available...")
                    break
            except Exception as e:
                _log(f"[ERROR] Unexpected error calling {endpoint} (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    time.sleep(1.0)
                else:
                    break

    _log("[ERROR] All configured LLM endpoints failed.")
    return None
