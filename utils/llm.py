"""
llm.py — Ollama LLM interface
Communicates with the locally running Ollama (llama3.2:latest) via subprocess.

Windows fixes applied:
  - Input sent as UTF-8 bytes (avoids cp1252 UnicodeDecodeError)
  - Output decoded as UTF-8 explicitly
  - ANSI/VT100 escape sequences stripped from output.
    Ollama emits cursor-control codes like ESC[2D and ESC[K when stdout is
    captured on Windows, garbling the response text. We strip them all.
"""

import re
import subprocess


MODEL = "llama3.2:latest"

# Robust ANSI/VT100 escape sequence stripper.
# Covers CSI sequences (ESC [ params letter), e.g. [2D [K [1;32m [?25l
# and bare Fe sequences (ESC @-_), plus carriage returns from progress lines.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b[@-_][0-9;]*|\r")


def _strip_ansi(text: str) -> str:
    """Remove all ANSI escape codes and carriage returns from text."""
    return _ANSI_RE.sub("", text)


def query_ollama(prompt: str, system_prompt: str = "") -> str:
    """
    Send a prompt to Ollama and return clean response text.
    Uses 'ollama run' via subprocess — no external API calls.
    """
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

    try:
        result = subprocess.run(
            ["ollama", "run", "--nowordwrap", MODEL],
            input=full_prompt.encode("utf-8"),  # avoid Windows cp1252 encoding
            capture_output=True,                 # capture raw bytes
            timeout=120,
        )

        if result.returncode != 0:
            error_msg = _strip_ansi(
                result.stderr.decode("utf-8", errors="replace")
            ).strip()
            return f"[LLM Error] {error_msg or 'Unknown error from Ollama'}"

        # Decode as UTF-8, strip ANSI codes, clean whitespace
        raw = result.stdout.decode("utf-8", errors="replace")
        return _strip_ansi(raw).strip()

    except FileNotFoundError:
        return "[LLM Error] Ollama not found. Please install Ollama and pull llama3.2:latest."
    except subprocess.TimeoutExpired:
        return "[LLM Error] Ollama response timed out after 120 seconds."
    except Exception as e:
        return f"[LLM Error] {str(e)}"
