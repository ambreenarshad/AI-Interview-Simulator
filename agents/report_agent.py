"""
report_agent.py — Final Report Generator Agent
Uses Ollama to produce a holistic, narrative final assessment
based on all Q&A records from the session.
"""

from utils.llm import query_ollama


SYSTEM_PROMPT = """You are a senior hiring manager and interview coach writing a final 
performance assessment. You are thorough, fair, and constructive."""


def generate_final_summary(report: dict) -> str:
    """
    Generate a natural-language summary of the entire interview session.

    Args:
        report: The dict returned by InterviewSession.build_final_report()

    Returns:
        A narrative paragraph summarizing overall performance.
    """
    avg = report["average_scores"]
    role = report["role"]
    interview_type = report["interview_type"]
    total_q = report["total_questions"]
    tier = report["performance_tier"]

    # Build a compact summary of each Q&A for the prompt
    qa_summary_lines = []
    for rec in report.get("qa_records", [])[:10]:  # limit to 10 to keep prompt manageable
        q_num = rec["question_number"]
        q_text = rec["question"][:100] + "..." if len(rec["question"]) > 100 else rec["question"]
        score = rec["evaluation"].get("overall", 0)
        qa_summary_lines.append(f"  Q{q_num}: {q_text}  → Score: {score}/10")

    qa_summary = "\n".join(qa_summary_lines)

    prompt = f"""Write a concise professional interview performance summary.

Candidate Details:
- Role: {role}
- Interview Type: {interview_type}
- Total Questions: {total_q}
- Performance Tier: {tier}

Average Scores:
- Clarity: {avg['clarity']}/10
- Relevance: {avg['relevance']}/10
- Depth: {avg['depth']}/10
- Structure: {avg['structure']}/10
- Overall: {avg['overall']}/10

Question-by-Question Overview:
{qa_summary}

Write a 3–4 sentence professional narrative summary. Mention the overall score level,
key patterns observed, and what the candidate should focus on going forward.
Be honest but encouraging. Do NOT repeat the numbers — interpret them in words."""

    summary = query_ollama(prompt, SYSTEM_PROMPT)
    return summary.strip() if summary else "Performance summary could not be generated."
