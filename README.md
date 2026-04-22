# 🤖 AI Interview Simulator
### Multi-Agent Evaluation · Adaptive Questioning · Voice Interface

A complete, GUI-based AI interview simulator powered by **Ollama (llama3.2:latest)** running locally.
No external APIs. No subscriptions. Fully offline-capable (except Google STT).

---

## 📁 Project Structure

```
ai_interview_simulator/
│
├── app.py                        # Main GUI application (entry point)
│
├── agents/
│   ├── interviewer_agent.py      # Generates interview questions via Ollama
│   ├── evaluator_agent.py        # Scores answers on 4 dimensions (1–10)
│   ├── feedback_agent.py         # Provides strengths, weaknesses, suggestions
│   ├── report_agent.py           # Generates final narrative summary
│   └── session.py                # Session manager + difficulty adapter
│
├── utils/
│   ├── llm.py                    # Ollama subprocess interface
│   └── voice.py                  # TTS (gTTS + pyttsx3) + STT (SpeechRecognition)
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Prerequisites

### 1. Install Ollama
```bash
# Linux / macOS
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: Download from https://ollama.ai
```

### 2. Pull the model
```bash
ollama pull llama3.2:latest
```

### 3. Make sure Ollama is running
```bash
ollama serve   # keep this running in a terminal
```

---

## 🚀 Installation

### Linux / Ubuntu
```bash
# System dependencies
sudo apt-get install python3-tk portaudio19-dev

# Python packages
pip install -r requirements.txt
```

### macOS
```bash
brew install portaudio
pip install -r requirements.txt
```

### Windows
```bash
# Install portaudio via conda or download binary
conda install pyaudio
pip install gTTS SpeechRecognition pyttsx3 pygame
```

---

## ▶️ Running the App

```bash
cd ai_interview_simulator
python app.py
```

That's it! The GUI will open automatically.

---

## 🎯 How to Use

1. **Welcome Screen**
   - Select your **Role** (e.g., Software Engineer, Data Scientist)
   - Select **Interview Type**: HR / Technical / Mixed
   - Toggle **Voice Mode** on/off (requires microphone)
   - Click **START INTERVIEW**

2. **Interview Screen**
   - Questions are displayed and spoken aloud (TTS)
   - Speak your answer when prompted (STT auto-records)
   - Or type manually and click Submit (if voice is off)
   - Use Skip to skip a question

3. **Feedback Screen** (after each answer)
   - See scores for Clarity, Relevance, Depth, Structure
   - Read strengths, weaknesses, suggestions
   - See an AI-improved version of your answer
   - Click Continue for next question

4. **Final Report** (after 8–10 questions)
   - See average scores across all dimensions
   - Read aggregated strengths and improvement areas
   - AI narrative summary of your performance
   - Save report as JSON

---

## 🤖 Agent Architecture

| Agent | Function | Model |
|-------|----------|-------|
| **Interviewer Agent** | Generates role-specific questions | llama3.2:latest |
| **Evaluator Agent** | Scores on Clarity, Relevance, Depth, Structure | llama3.2:latest |
| **Feedback Agent** | Strengths, weaknesses, improved answer | llama3.2:latest |
| **Report Agent** | Narrative session summary | llama3.2:latest |
| **Difficulty Adapter** | score >7 → hard, <4 → easy, else medium | Rule-based |

---

## 📊 Evaluation Dimensions

| Dimension | What it measures |
|-----------|-----------------|
| **Clarity (1–10)** | Is the answer clear and understandable? |
| **Relevance (1–10)** | Does it address the question directly? |
| **Depth (1–10)** | Is it detailed and substantive? |
| **Structure (1–10)** | Does it follow STAR (Situation, Task, Action, Result)? |

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `Ollama not found` | Run `ollama serve` and ensure it's in PATH |
| `No module named tkinter` | `sudo apt-get install python3-tk` |
| `pyaudio build failed` | `sudo apt-get install portaudio19-dev` then reinstall |
| `No speech detected` | Check microphone permissions; speak clearly |
| Questions are slow | Normal — llama3.2 generation can take 10–30s locally |
| STT not working | Requires internet for Google Web Speech; use text mode |

---

## 💾 Session Reports

Reports are saved as JSON files in the project directory:
```
interview_report_20240101_120000.json
```

Contains: all Q&A pairs, scores per question, feedback, averages, and AI summary.

---

## 🔑 Key Technical Decisions

- **Ollama via subprocess** — `ollama run llama3.2:latest` called via `subprocess.run()`
- **TTS**: gTTS (converts text → MP3) played with pygame; pyttsx3 as offline fallback  
- **STT**: `SpeechRecognition` library with Google Web Speech (free, no API key needed)
- **GUI**: Pure tkinter — no extra GUI framework dependencies
- **Threading**: All LLM calls and voice I/O run in background threads to keep UI responsive
- **Session state**: Stored in `InterviewSession` dataclass (JSON-serializable)
