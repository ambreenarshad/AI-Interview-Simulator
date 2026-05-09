"""
utils/llm.py
============
Unified Ollama interface supporting multiple models.

Model assignments (justified by benchmark results):
  - Question generation  → llama3.2  (ROUGE-L 0.501, best answer quality)
  - Answer evaluation    → gemma2:2b (σ=0.155 lowest consistency, 66.9s fastest)
  - Feedback generation  → gemma2:2b (consistent, reliable structured output)
  - Final report summary → gemma2:2b (stable, good overall score 8.24)

Benchmark evidence (30 samples, 3 consistency runs each):
  Model       ROUGE-L   Latency   Consistency-σ   Overall
  ──────────────────────────────────────────────────────
  llama3.2    0.5011    100.5s    0.3240          7.75
  gemma2      0.4505     66.9s    0.1551          8.24   ← evaluator
  phi3        0.1070    179.1s    0.7627          5.81   ← excluded
  mistral     0.2756    206.9s    0.2965          8.52   ← too slow
"""

import re
import subprocess

# ── Model assignments ──────────────────────────────────────────────────────────
MODEL_QUESTIONER = "llama3.2:latest"  # best ROUGE-L = most relevant questions
MODEL_EVALUATOR  = "gemma2:2b"        # lowest consistency σ + fastest latency
MODEL_FEEDBACK   = "gemma2:2b"        # same — consistent structured output
MODEL_REPORTER   = "gemma2:2b"        # same — reliable summary generation

# Legacy default (used if query_ollama called without model arg)
MODEL_DEFAULT    = "llama3.2:latest"

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b[@-_][0-9;]*|\r")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def query_ollama(
    prompt: str,
    system_prompt: str = "",
    model: str | None = None,
    timeout: int = 120,
) -> str:
    """
    Send a prompt to Ollama and return the stripped text response.

    Parameters
    ----------
    prompt        : the user prompt
    system_prompt : optional system/role context prepended to the prompt
    model         : which Ollama model to use (defaults to MODEL_DEFAULT)
    timeout       : seconds before subprocess timeout (default 120)

    Returns
    -------
    str — model response, or "[LLM Error] ..." on failure
    """
    model = model or MODEL_DEFAULT
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

    try:
        result = subprocess.run(
            ["ollama", "run", "--nowordwrap", model],
            input=full_prompt.encode("utf-8"),
            capture_output=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            error_msg = _strip_ansi(
                result.stderr.decode("utf-8", errors="replace")
            ).strip()
            return f"[LLM Error] {error_msg or 'Unknown error from Ollama'}"
        raw = result.stdout.decode("utf-8", errors="replace")
        return _strip_ansi(raw).strip()

    except FileNotFoundError:
        return (
            "[LLM Error] Ollama not found. "
            "Install Ollama and run: ollama pull llama3.2:latest && ollama pull gemma2:2b"
        )
    except subprocess.TimeoutExpired:
        return f"[LLM Error] Ollama timed out after {timeout}s."
    except Exception as e:
        return f"[LLM Error] {str(e)}"


# ── Convenience wrappers — named by role so each agent imports cleanly ─────────

def query_questioner(prompt: str, system_prompt: str = "") -> str:
    """Use llama3.2 for question generation (best ROUGE-L)."""
    return query_ollama(prompt, system_prompt, model=MODEL_QUESTIONER)


def query_evaluator(prompt: str, system_prompt: str = "") -> str:
    """Use gemma2 for answer evaluation (lowest σ, fastest)."""
    return query_ollama(prompt, system_prompt, model=MODEL_EVALUATOR)


def query_feedback(prompt: str, system_prompt: str = "") -> str:
    """Use gemma2 for feedback generation (consistent structured output)."""
    return query_ollama(prompt, system_prompt, model=MODEL_FEEDBACK)


def query_reporter(prompt: str, system_prompt: str = "") -> str:
    """Use gemma2 for final report summary (reliable, stable)."""
    return query_ollama(prompt, system_prompt, model=MODEL_REPORTER)