"""
interviewer_agent.py — Interviewer Agent
Generates contextual interview questions using Ollama (llama3.2:latest).
Supports role-based, difficulty-adapted, and type-specific question generation.
"""

from utils.llm import query_ollama


SYSTEM_PROMPT = """You are an expert interviewer with 15+ years of experience conducting 
technical and HR interviews at top tech companies. You ask clear, relevant, and 
challenging questions tailored to the candidate's role and experience level."""


def generate_question(
    role: str,
    difficulty: str,
    interview_type: str,
    previous_topics: list[str] = None,
    question_number: int = 1,
) -> str:
    """
    Generate a single interview question.

    Args:
        role: Job role (e.g., 'Software Engineer', 'Data Scientist')
        difficulty: 'easy', 'medium', or 'hard'
        interview_type: 'HR', 'Technical', or 'Mixed'
        previous_topics: List of topics already covered (to avoid repetition)
        question_number: Current question number in the session

    Returns:
        A single interview question as a string.
    """
    avoided = ""
    if previous_topics:
        topics_str = ", ".join(previous_topics[-5:])  # only last 5 to keep prompt short
        avoided = f"\nAVOID repeating these topics already covered: {topics_str}"

    type_instruction = {
        "HR": "Focus on behavioral, situational, cultural fit, and soft skills questions.",
        "Technical": f"Focus on technical skills, problem-solving, and domain knowledge relevant to {role}.",
        "Mixed": "Alternate between behavioral/HR questions and technical skill questions.",
    }.get(interview_type, "Mix of HR and technical questions.")

    difficulty_guide = {
        "easy": "Ask a straightforward, entry-level question.",
        "medium": "Ask a moderately challenging question that requires practical experience.",
        "hard": "Ask a complex, senior-level question requiring deep expertise and critical thinking.",
    }.get(difficulty, "Ask a moderately challenging question.")

    prompt = f"""Generate ONE interview question for the following context:

Role: {role}
Interview Type: {interview_type}
Difficulty: {difficulty}
Question Number: {question_number}

Instructions:
- {type_instruction}
- {difficulty_guide}
- Ask ONE clear, focused question only.
- Do NOT include any preamble, numbering, or explanation.
- Do NOT include the answer.
- The question should be conversational and natural.{avoided}

Output ONLY the question text, nothing else."""

    response = query_ollama(prompt, SYSTEM_PROMPT)

    # Clean up any accidental prefixes like "Question:" or numbering
    question = response.strip()
    for prefix in ["Question:", "Q:", "1.", "2.", "3.", "-", "*", "•"]:
        if question.startswith(prefix):
            question = question[len(prefix):].strip()

    return question if question else f"Tell me about your experience as a {role}."
