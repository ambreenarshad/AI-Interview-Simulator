"""
interviewer_agent.py — Generates interview questions via Ollama
"""

from utils.llm import query_ollama

SYSTEM_PROMPT = """You are an expert interviewer with 15+ years of experience. 
You ask clear, targeted, and professionally relevant questions."""


def generate_question(
    role: str,
    difficulty: str,
    interview_type: str,
    previous_topics: list[str] = None,
    question_number: int = 1,
) -> str:
    avoided = ""
    if previous_topics:
        topics_str = ", ".join(previous_topics[-5:])
        avoided = f"\nDO NOT repeat these already-covered topics: {topics_str}"

    type_instruction = {
        "HR": "Ask a behavioral, situational, or cultural fit question.",
        "Technical": f"Ask a technical, problem-solving question for a {role}.",
        "Mixed": "Alternate between behavioral and technical questions.",
    }.get(interview_type, "Ask a relevant interview question.")

    difficulty_guide = {
        "easy": "Make it entry-level and straightforward.",
        "medium": "Make it moderately challenging, requiring real experience.",
        "hard": "Make it senior-level, requiring deep expertise.",
    }.get(difficulty, "Make it moderately challenging.")

    prompt = f"""Generate ONE interview question for this context:

Role: {role}
Type: {interview_type} | Difficulty: {difficulty} | Question #{question_number}

Rules:
- {type_instruction}
- {difficulty_guide}
- Output ONLY the question — no numbering, no preamble, no explanation.
- Be concise and conversational.{avoided}"""

    response = query_ollama(prompt, SYSTEM_PROMPT)
    question = response.strip()

    for prefix in ["Question:", "Q:", "1.", "2.", "3.", "-", "*", "•"]:
        if question.startswith(prefix):
            question = question[len(prefix):].strip()

    return question or f"Tell me about your experience as a {role}."
