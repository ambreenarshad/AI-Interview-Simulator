"""
agents/report_agent.py
======================
Generates the final interview narrative summary using gemma2:2b.

Why gemma2 for report generation?
──────────────────────────────────
The final report summary requires:
  1. Consistent quality — report is the last thing the user sees
  2. Fast generation   — user has been waiting through the whole interview
  3. Coherent prose    — 3-4 paragraph narrative, not just lists

gemma2 delivers on all three (σ=0.155, 66.9s avg, overall 8.24/10).
"""

from utils.llm import query_reporter   # gemma2:2b

SYSTEM_PROMPT = """You are a senior hiring manager writing a final interview assessment.
Be concise, fair, and constructive."""


def generate_final_summary(report: dict) -> str:
    """
    Generate a 3-4 sentence narrative summary of the full interview.

    Parameters
    ----------
    report : dict from session.build_final_report()

    Returns
    -------
    str — narrative summary suitable for the final report screen
    """
    avg          = report["average_scores"]
    role         = report["role"]
    interview_type = report["interview_type"]
    total_q      = report["total_questions"]
    tier         = report["performance_tier"]

    qa_lines = []
    for rec in report.get("qa_records", [])[:8]:
        q_text = rec["question"][:80] + "..." if len(rec["question"]) > 80 else rec["question"]
        score  = rec["evaluation"].get("overall", 0)
        skipped = rec["evaluation"].get("skipped", False)
        status = "skipped" if skipped else f"{score}/10"
        qa_lines.append(f"  Q{rec['question_number']}: {q_text} → {status}")

    prompt = f"""Write a concise 3-4 sentence professional interview summary.

Role: {role} | Type: {interview_type} | Questions: {total_q} | Tier: {tier}
Scores — Clarity: {avg['clarity']} | Relevance: {avg['relevance']} | Depth: {avg['depth']} | Structure: {avg['structure']} | Overall: {avg['overall']}/10

{chr(10).join(qa_lines)}

Instructions:
- Describe performance patterns in words, do NOT list raw numbers.
- Be encouraging but honest about areas to improve.
- 3-4 sentences maximum."""

    # Uses gemma2:2b — consistent narrative generation, σ=0.155
    summary = query_reporter(prompt, SYSTEM_PROMPT)
    return summary.strip() or "Performance summary could not be generated."