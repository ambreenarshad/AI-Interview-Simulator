"""
main.py — FastAPI backend for AI Interview Simulator
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import uuid
import json
from datetime import datetime

from agents.interviewer_agent import generate_question
from agents.evaluator_agent import evaluate_answer
from agents.feedback_agent import generate_feedback
from agents.report_agent import generate_final_summary
from agents.session import InterviewSession, adapt_difficulty

app = FastAPI(title="AI Interview Simulator")

# In-memory session store
sessions: dict[str, InterviewSession] = {}

TOTAL_QUESTIONS = 8

# ── Models ─────────────────────────────────────────────────────────────────────

class StartRequest(BaseModel):
    role: str
    interview_type: str  # HR, Technical, Mixed

class SubmitAnswerRequest(BaseModel):
    session_id: str
    answer: str

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse("static/index.html")

@app.post("/api/start")
async def start_interview(req: StartRequest):
    """Create a new interview session and return the first question."""
    session_id = str(uuid.uuid4())
    session = InterviewSession(role=req.role, interview_type=req.interview_type)
    sessions[session_id] = session

    # Generate first question
    question = generate_question(
        role=req.role,
        difficulty="medium",
        interview_type=req.interview_type,
        previous_topics=[],
        question_number=1,
    )
    session.current_question = question
    session.question_number = 1

    return {
        "session_id": session_id,
        "question": question,
        "question_number": 1,
        "total_questions": TOTAL_QUESTIONS,
        "difficulty": "medium",
    }


@app.post("/api/submit")
async def submit_answer(req: SubmitAnswerRequest):
    """Evaluate answer, generate feedback, and return next question if applicable."""
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    question = session.current_question
    answer = req.answer.strip()

    # Evaluate
    evaluation = evaluate_answer(question, answer)
    feedback = generate_feedback(question, answer, evaluation)

    # Record
    session.add_record(question, answer, evaluation, feedback)

    is_last = session.question_number >= TOTAL_QUESTIONS

    result = {
        "evaluation": evaluation,
        "feedback": feedback,
        "question_number": session.question_number,
        "is_last": is_last,
    }

    if not is_last:
        # Generate next question
        next_q = generate_question(
            role=session.role,
            difficulty=session.current_difficulty,
            interview_type=session.interview_type,
            previous_topics=session.questions_asked[-5:],
            question_number=session.question_number + 1,
        )
        session.current_question = next_q
        session.question_number += 1
        result["next_question"] = next_q
        result["next_question_number"] = session.question_number
        result["next_difficulty"] = session.current_difficulty

    return result


@app.get("/api/report/{session_id}")
async def get_report(session_id: str):
    """Generate and return the final report."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    report = session.build_final_report()
    summary = generate_final_summary(report)
    report["ai_summary"] = summary
    return report


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    sessions.pop(session_id, None)
    return {"ok": True}


# ── Static files ───────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")
