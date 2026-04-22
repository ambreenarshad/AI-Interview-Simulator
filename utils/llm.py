"""
llm.py — Ollama LLM interface
Communicates with the locally running Ollama (llama3.2:latest) via subprocess.
"""

import subprocess
import json


MODEL = "llama3.2:latest"


def query_ollama(prompt: str, system_prompt: str = "") -> str:
    """
    Send a prompt to Ollama and return the response text.
    Uses 'ollama run' via subprocess — no external API calls.
    """
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

    try:
        result = subprocess.run(
            ["ollama", "run", MODEL],
            input=full_prompt,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            error_msg = result.stderr.strip()
            return f"[LLM Error] {error_msg or 'Unknown error from Ollama'}"

        return result.stdout.strip()

    except FileNotFoundError:
        return "[LLM Error] Ollama not found. Please install Ollama and pull llama3.2:latest."
    except subprocess.TimeoutExpired:
        return "[LLM Error] Ollama response timed out after 120 seconds."
    except Exception as e:
        return f"[LLM Error] {str(e)}"
