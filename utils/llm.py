"""
llm.py — Ollama interface via subprocess
"""

import re
import subprocess

MODEL = "llama3.2:latest"
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b[@-_][0-9;]*|\r")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def query_ollama(prompt: str, system_prompt: str = "") -> str:
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
    try:
        result = subprocess.run(
            ["ollama", "run", "--nowordwrap", MODEL],
            input=full_prompt.encode("utf-8"),
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0:
            error_msg = _strip_ansi(result.stderr.decode("utf-8", errors="replace")).strip()
            return f"[LLM Error] {error_msg or 'Unknown error from Ollama'}"
        raw = result.stdout.decode("utf-8", errors="replace")
        return _strip_ansi(raw).strip()
    except FileNotFoundError:
        return "[LLM Error] Ollama not found. Please install Ollama and run: ollama pull llama3.2:latest"
    except subprocess.TimeoutExpired:
        return "[LLM Error] Ollama timed out after 120 seconds."
    except Exception as e:
        return f"[LLM Error] {str(e)}"
