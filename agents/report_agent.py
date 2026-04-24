"""
report_agent.py — Generates final narrative summary
"""

from utils.llm import query_ollama

SYSTEM_PROMPT = """You are a senior hiring manager writing a final interview assessment. 
Be concise, fair, and constructive."""


def generate_final_summary(report: dict) -> str:
    avg = report["average_scores"]
    role = report["role"]
    interview_type = report["interview_type"]
    total_q = report["total_questions"]
    tier = report["performance_tier"]

    qa_lines = []
    for rec in report.get("qa_records", [])[:8]:
        q_text = rec["question"][:80] + "..." if len(rec["question"]) > 80 else rec["question"]
        score = rec["evaluation"].get("overall", 0)
        qa_lines.append(f"  Q{rec['question_number']}: {q_text} → {score}/10")

    prompt = f"""Write a concise 3-4 sentence professional interview summary.

Role: {role} | Type: {interview_type} | Questions: {total_q} | Tier: {tier}
Avg: Clarity {avg['clarity']} | Relevance {avg['relevance']} | Depth {avg['depth']} | Structure {avg['structure']} | Overall {avg['overall']}/10

{chr(10).join(qa_lines)}

Interpret scores in words. Be encouraging but honest. Do NOT list numbers — describe patterns."""

    summary = query_ollama(prompt, SYSTEM_PROMPT)
    return summary.strip() or "Performance summary could not be generated."
