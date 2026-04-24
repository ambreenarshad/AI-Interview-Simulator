"""
session.py — Session Manager and Difficulty Adapter
"""

import json
from datetime import datetime


def adapt_difficulty(current_score: float) -> str:
    if current_score > 7:
        return "hard"
    elif current_score < 4:
        return "easy"
    else:
        return "medium"


class InterviewSession:
    def __init__(self, role: str, interview_type: str):
        self.role = role
        self.interview_type = interview_type
        self.start_time = datetime.now().isoformat()
        self.end_time = None
        self.current_difficulty = "medium"
        self.question_number = 0
        self.current_question = ""
        self.questions_asked: list[str] = []
        self.qa_records: list[dict] = []

    def add_record(self, question: str, answer: str, evaluation: dict, feedback: dict):
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
        self.current_difficulty = adapt_difficulty(evaluation.get("overall", 5.0))

    def get_average_scores(self) -> dict:
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
        return [r["evaluation"].get("overall", 0) for r in self.qa_records]

    def build_final_report(self) -> dict:
        avg = self.get_average_scores()
        trend = self.get_score_trend()
        self.end_time = datetime.now().isoformat()

        all_strengths = [r["feedback"].get("strengths", "") for r in self.qa_records if r["feedback"].get("strengths")]
        all_weaknesses = [r["feedback"].get("weaknesses", "") for r in self.qa_records if r["feedback"].get("weaknesses")]
        all_suggestions = [r["feedback"].get("suggestions", "") for r in self.qa_records if r["feedback"].get("suggestions")]

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
            "total_questions": len(self.qa_records),
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
