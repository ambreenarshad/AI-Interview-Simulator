# 🤖 AI Interview Simulator — Web App

A FastAPI + vanilla JS web application for AI-powered mock interviews using Ollama locally.

---

## 🚀 Quick Start (3 steps)

### Step 1 — Start Ollama
```bash
ollama serve
# In another terminal:
ollama pull llama3.2:latest
```

### Step 2 — Install Python dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run the web server
```bash
uvicorn main:app --reload --port 8000
```

Then open: **http://localhost:8000**

---

## 📁 Project Structure

```
interview_app/
├── main.py                  # FastAPI app + API routes
├── requirements.txt
├── README.md
│
├── agents/
│   ├── interviewer_agent.py # Generates questions
│   ├── evaluator_agent.py   # Scores answers (Clarity/Relevance/Depth/Structure)
│   ├── feedback_agent.py    # Strengths, weaknesses, suggestions
│   ├── report_agent.py      # Final narrative summary
│   └── session.py           # Session state + difficulty adapter
│
├── utils/
│   └── llm.py               # Ollama subprocess interface
│
└── static/
    └── index.html           # Full single-page frontend
```

---

## 🎯 Features

| Feature | Details |
|---------|---------|
| **8 Questions** | Exactly 8 per session, no more no less |
| **90s Timer** | Visible countdown per question, auto-submits on expiry |
| **Voice Input** | Browser Web Speech API (Chrome recommended) |
| **Adaptive Difficulty** | Score >7 → Hard, <4 → Easy, else Medium |
| **4-Dim Scoring** | Clarity · Relevance · Depth · Structure |
| **Real-time Feedback** | After every answer |
| **Final Report** | Summary + trend + AI narrative |
| **Save Report** | Download JSON |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/start` | Start session, get Q1 |
| POST | `/api/submit` | Submit answer, get evaluation + next Q |
| GET | `/api/report/{session_id}` | Get final report |
| DELETE | `/api/session/{session_id}` | Clean up session |

---

## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| `Connection refused` | Run `ollama serve` |
| `LLM Error: Ollama not found` | Add ollama to PATH |
| Voice not working | Use Chrome browser |
| Slow responses | Normal — llama3.2 takes 10-30s locally |
| Port in use | `uvicorn main:app --port 8001` |
