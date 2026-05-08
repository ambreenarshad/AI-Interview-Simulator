"""
agents/model_registry.py — Unified wrapper for all three LLMs.
Lets any agent swap models by passing model_id=...
"""

import re
import subprocess

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b[@-_][0-9;]*|\r")

# ── Registry ───────────────────────────────────────────────────────────────────
MODELS = {
    "llama":  "llama3.2:latest",
    "gemma":  "gemma2:2b",
    "phi3":   "phi3:latest",
    "mistral": "mistral:latest",
}

DEFAULT_MODEL = "llama"


def list_models() -> list[str]:
    return list(MODELS.keys())


def resolve(model_id: str) -> str:
    """Return the full Ollama model string for a short model_id."""
    if model_id not in MODELS:
        raise ValueError(f"Unknown model '{model_id}'. Choose from: {list(MODELS.keys())}")
    return MODELS[model_id]


def query(prompt: str, model_id: str = DEFAULT_MODEL,
          system_prompt: str = "", timeout: int = 120) -> str:
    """
    Send a prompt to an Ollama model and return the response string.
    Falls back gracefully on errors.
    """
    model_name = resolve(model_id)
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

    try:
        result = subprocess.run(
            ["ollama", "run", "--nowordwrap", model_name],
            input=full_prompt.encode("utf-8"),
            capture_output=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            err = _ANSI.sub("", result.stderr.decode("utf-8", errors="replace")).strip()
            return f"[ERROR] {err or 'Ollama returned non-zero exit code'}"
        raw = result.stdout.decode("utf-8", errors="replace")
        return _ANSI.sub("", raw).strip()

    except FileNotFoundError:
        return "[ERROR] ollama not found. Run: ollama serve"
    except subprocess.TimeoutExpired:
        return f"[ERROR] Timeout after {timeout}s"
    except Exception as e:
        return f"[ERROR] {e}"