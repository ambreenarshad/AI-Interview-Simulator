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

from agents.interviewer_agent import generate_question, estimate_time_limit
from agents.evaluator_agent import evaluate_answer
from agents.feedback_agent import generate_feedback
from agents.report_agent import generate_final_summary
from agents.session import InterviewSession, adapt_difficulty

app = FastAPI(title="AI Interview Simulator")

# In-memory session store
sessions: dict[str, InterviewSession] = {}

# ── Models ─────────────────────────────────────────────────────────────────────

class StartRequest(BaseModel):
    role: str
    interview_type: str  # HR, Technical, Mixed
    num_questions: int = 5  # slider value 3-15

class SubmitAnswerRequest(BaseModel):
    session_id: str
    answer: str

class SkipQuestionRequest(BaseModel):
    session_id: str

class ExtendRequest(BaseModel):
    session_id: str
    extra_questions: int

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse("static/index.html")

@app.post("/api/start")
async def start_interview(req: StartRequest):
    """Create a new interview session and return the first question."""
    session_id = str(uuid.uuid4())
    session = InterviewSession(role=req.role, interview_type=req.interview_type)
    total = max(3, min(req.num_questions, 15))
    session.total_questions = total
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

    # ── CHANGE: compute dynamic time limit for the first question ──
    time_limit = estimate_time_limit(question, "medium")

    return {
        "session_id": session_id,
        "question": question,
        "question_number": 1,
        "total_questions": total,
        "difficulty": "medium",
        "time_limit": time_limit,          # NEW
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

    is_last = session.question_number >= session.total_questions

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

        # ── CHANGE: compute dynamic time limit for the next question ──
        next_time_limit = estimate_time_limit(next_q, session.current_difficulty)

        result["next_question"] = next_q
        result["next_question_number"] = session.question_number
        result["next_difficulty"] = session.current_difficulty
        result["next_time_limit"] = next_time_limit   # NEW

    return result


@app.post("/api/skip")
async def skip_question(req: SkipQuestionRequest):
    """Skip current question without evaluation — record it as skipped and move on."""
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    question = session.current_question

    skipped_evaluation = {
        "clarity": 0, "relevance": 0, "depth": 0, "structure": 0, "overall": 0.0,
        "explanation": "Question was skipped.",
        "raw_response": "",
        "skipped": True,
    }
    skipped_feedback = {
        "strengths": "N/A — question was skipped.",
        "weaknesses": "Always attempt every question; even a partial answer scores better than silence.",
        "suggestions": "Try to answer every question. Interviewers notice when candidates skip. A brief, honest attempt is always better than no response.",
        "improved_answer": "N/A",
        "raw_response": "",
    }
    session.add_record(question, "", skipped_evaluation, skipped_feedback)

    is_last = session.question_number >= session.total_questions

    result = {
        "question_number": session.question_number,
        "is_last": is_last,
        "skipped": True,
    }

    if not is_last:
        next_q = generate_question(
            role=session.role,
            difficulty=session.current_difficulty,
            interview_type=session.interview_type,
            previous_topics=session.questions_asked[-5:],
            question_number=session.question_number + 1,
        )
        session.current_question = next_q
        session.question_number += 1

        # ── CHANGE: compute dynamic time limit for the next question ──
        next_time_limit = estimate_time_limit(next_q, session.current_difficulty)

        result["next_question"] = next_q
        result["next_question_number"] = session.question_number
        result["next_difficulty"] = session.current_difficulty
        result["next_time_limit"] = next_time_limit   # NEW

    return result


@app.post("/api/extend")
async def extend_interview(req: ExtendRequest):
    """Extend the session with more questions after the original set is done."""
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    extra = max(1, min(req.extra_questions, 20))
    session.total_questions = session.question_number + extra

    next_q = generate_question(
        role=session.role,
        difficulty=session.current_difficulty,
        interview_type=session.interview_type,
        previous_topics=session.questions_asked[-5:],
        question_number=session.question_number + 1,
    )
    session.current_question = next_q
    session.question_number += 1

    # ── CHANGE: compute dynamic time limit ──
    next_time_limit = estimate_time_limit(next_q, session.current_difficulty)

    return {
        "total_questions": session.total_questions,
        "next_question": next_q,
        "next_question_number": session.question_number,
        "next_difficulty": session.current_difficulty,
        "next_time_limit": next_time_limit,            # NEW
    }


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