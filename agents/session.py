"""
session.py — Session Manager and Difficulty Adapter
Manages interview session state, scoring history, and adaptive difficulty logic.
"""

import json
from datetime import datetime


# ─── Difficulty Adapter ────────────────────────────────────────────────────────

def adapt_difficulty(current_score: float) -> str:
    """
    Adapt question difficulty based on the candidate's last overall score.

    Rules:
        score > 7  → hard
        score < 4  → easy
        else       → medium
    """
    if current_score > 7:
        return "hard"
    elif current_score < 4:
        return "easy"
    else:
        return "medium"


# ─── Session Manager ──────────────────────────────────────────────────────────

class InterviewSession:
    """Holds all data for a single interview session."""

    def __init__(self, role: str, interview_type: str):
        self.role = role
        self.interview_type = interview_type
        self.start_time = datetime.now().isoformat()
        self.end_time = None

        self.current_difficulty = "medium"  # start at medium
        self.question_number = 0
        self.questions_asked: list[str] = []
        self.topics_covered: list[str] = []

        # Each entry: {question, answer, evaluation, feedback}
        self.qa_records: list[dict] = []

    # ── Recording ─────────────────────────────────────────────────────────────

    def add_record(
        self,
        question: str,
        answer: str,
        evaluation: dict,
        feedback: dict,
    ):
        """Record a completed Q&A exchange with evaluation and feedback."""
        self.question_number += 1
        self.questions_asked.append(question)

        record = {
            "question_number": self.question_number,
            "difficulty": self.current_difficulty,
            "question": question,
            "answer": answer,
            "evaluation": evaluation,
            "feedback": feedback,
        }
        self.qa_records.append(record)

        # Update difficulty for next question
        self.current_difficulty = adapt_difficulty(evaluation.get("overall", 5.0))

    # ── Aggregated Statistics ─────────────────────────────────────────────────

    def get_average_scores(self) -> dict:
        """Compute average scores across all recorded answers."""
        if not self.qa_records:
            return {"clarity": 0, "relevance": 0, "depth": 0, "structure": 0, "overall": 0}

        totals = {"clarity": 0, "relevance": 0, "depth": 0, "structure": 0, "overall": 0}
        count = len(self.qa_records)

        for record in self.qa_records:
            ev = record["evaluation"]
            for key in totals:
                totals[key] += ev.get(key, 0)

        return {k: round(v / count, 2) for k, v in totals.items()}

    def get_score_trend(self) -> list[float]:
        """Return list of overall scores per question for charting."""
        return [r["evaluation"].get("overall", 0) for r in self.qa_records]

    # ── Final Report Data ─────────────────────────────────────────────────────

    def build_final_report(self) -> dict:
        """Compile a final report from all session data."""
        avg = self.get_average_scores()
        trend = self.get_score_trend()
        self.end_time = datetime.now().isoformat()

        # Aggregate strengths and weaknesses across all feedback
        all_strengths = [r["feedback"].get("strengths", "") for r in self.qa_records if r["feedback"].get("strengths")]
        all_weaknesses = [r["feedback"].get("weaknesses", "") for r in self.qa_records if r["feedback"].get("weaknesses")]
        all_suggestions = [r["feedback"].get("suggestions", "") for r in self.qa_records if r["feedback"].get("suggestions")]

        # Determine performance tier
        overall = avg["overall"]
        if overall >= 8:
            tier = "Excellent 🏆"
            tier_msg = "Outstanding performance! You're well-prepared for this role."
        elif overall >= 6:
            tier = "Good 👍"
            tier_msg = "Solid performance with room to refine a few areas."
        elif overall >= 4:
            tier = "Average 📈"
            tier_msg = "Decent foundation — focus on depth and structure to stand out."
        else:
            tier = "Needs Improvement 💪"
            tier_msg = "Keep practicing! Structured preparation will help significantly."

        return {
            "role": self.role,
            "interview_type": self.interview_type,
            "total_questions": self.question_number,
            "average_scores": avg,
            "score_trend": trend,
            "performance_tier": tier,
            "tier_message": tier_msg,
            "overall_strengths": all_strengths,
            "overall_weaknesses": all_weaknesses,
            "overall_suggestions": all_suggestions,
            "qa_records": self.qa_records,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }

    def to_json(self) -> str:
        """Serialize session to JSON string for saving."""
        return json.dumps(self.build_final_report(), indent=2)

    def save_to_file(self, filepath: str):
        """Save session report to a JSON file."""
        with open(filepath, "w") as f:
            f.write(self.to_json())
