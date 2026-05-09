"""
agents/interviewer_agent.py
===========================
Generates interview questions using llama3.2:latest.

Why llama3.2 for question generation?
──────────────────────────────────────
Benchmark results (30 samples, ROUGE-L measures answer relevance to reference):

  Model       ROUGE-L   Interpretation
  ─────────────────────────────────────────────────────────────────
  llama3.2    0.5011    Highest — generates most contextually relevant content
  gemma2      0.4505    Good but 10% lower relevance than llama
  mistral     0.2756    High variance (std=0.308), unreliable
  phi3        0.1070    Poor — consistently low quality

llama3.2 produces the highest ROUGE-L score, meaning its outputs have
the greatest n-gram overlap with reference answers — indicating better
domain coverage and contextual relevance for question generation.
"""

from utils.llm import query_questioner, query_ollama   # llama3.2

SYSTEM_PROMPT = """You are an expert interviewer with 15+ years of experience.
You ask clear, targeted, and professionally relevant questions."""


def estimate_time_limit(question: str, difficulty: str) -> int:
    """
    Estimate answer time in seconds based on question complexity.
    Range: 60-180 seconds.

    Formula:
      base      = difficulty level base time
      +length   = proportional to question word count (longer Q = more context needed)
      +keywords = +10s per complexity keyword detected
    """
    base = {"easy": 60, "medium": 90, "hard": 120}.get(difficulty, 90)

    word_count = len(question.split())
    if word_count > 30:
        base += 20
    elif word_count > 20:
        base += 10

    complex_keywords = [
        "design", "architect", "compare", "tradeoff", "trade-off",
        "explain", "walk me through", "describe", "why", "how would you",
        "give an example", "tell me about a time", "scenario", "situation",
    ]
    matches = sum(1 for kw in complex_keywords if kw in question.lower())
    base += matches * 10

    return max(60, min(180, base))


def generate_question(
    role: str,
    difficulty: str,
    interview_type: str,
    previous_topics: list[str] = None,
    question_number: int = 1,
) -> str:
    """
    Generate a single interview question using llama3.2 (best ROUGE-L: 0.501).

    Parameters
    ----------
    role             : target job role
    difficulty       : easy | medium | hard
    interview_type   : HR | Technical | Mixed
    previous_topics  : list of already-asked questions (avoid repetition)
    question_number  : position in the interview sequence

    Returns
    -------
    str — the generated question
    """
    avoided = ""
    if previous_topics:
        topics_str = ", ".join(previous_topics[-5:])
        avoided = f"\nDO NOT repeat these already-covered topics: {topics_str}"

    type_instruction = {
        "HR":        "Ask a behavioral, situational, or cultural fit question.",
        "Technical": f"Ask a technical, problem-solving question for a {role}.",
        "Mixed":     "Alternate between behavioral and technical questions.",
    }.get(interview_type, "Ask a relevant interview question.")

    difficulty_guide = {
        "easy":   "Make it entry-level and straightforward.",
        "medium": "Make it moderately challenging, requiring real experience.",
        "hard":   "Make it senior-level, requiring deep expertise.",
    }.get(difficulty, "Make it moderately challenging.")

    prompt = f"""Generate ONE interview question for this context:

Role: {role}
Type: {interview_type} | Difficulty: {difficulty} | Question #{question_number}

Rules:
- {type_instruction}
- {difficulty_guide}
- Output ONLY the question — no numbering, no preamble, no explanation.
- Be concise and conversational.{avoided}"""

    # Uses llama3.2 — chosen for highest ROUGE-L (0.501) in benchmark
    response = query_questioner(prompt, SYSTEM_PROMPT)
    question = response.strip()

    for prefix in ["Question:", "Q:", "1.", "2.", "3.", "-", "*", "•"]:
        if question.startswith(prefix):
            question = question[len(prefix):].strip()

    return question or f"Tell me about your experience as a {role}."