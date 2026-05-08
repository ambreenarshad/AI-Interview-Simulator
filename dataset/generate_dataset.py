"""
generate_dataset.py — Generates a custom interview Q&A dataset using Ollama.

Usage:
    python generate_dataset.py

Output:
    data/processed/interview_dataset.json   ← main dataset
    data/processed/dataset_stats.json       ← statistics for your paper

This creates a UNIQUE dataset (good for bonus marks in the rubric).
Estimated time: ~20-40 minutes depending on your machine.
"""

import json
import time
import subprocess
import re
import random
from pathlib import Path
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────────

MODEL = "llama3.2:latest"   # change to mistral or gemma if preferred

ROLES = [
    "Software Engineer",
    "Data Scientist",
    "Machine Learning Engineer",
    "Frontend Developer",
    "Backend Developer",
    "DevOps Engineer",
    "Product Manager",
    "UX Designer",
    "Data Analyst",
    "Cybersecurity Analyst",
]

INTERVIEW_TYPES = ["HR", "Technical"]   # Mixed is a combo of these two

DIFFICULTIES = ["easy", "medium", "hard"]

# How many Q&A pairs to generate per (role, type, difficulty) combination.
# 10 roles × 2 types × 3 difficulties × SAMPLES_PER_COMBO = total dataset size
# At 2: 10×2×3×2 = 120 samples  ← fast, good enough
# At 4: 10×2×3×4 = 240 samples  ← better for the paper
SAMPLES_PER_COMBO = 2

OUTPUT_DIR = Path("data/processed")
OUTPUT_PATH = OUTPUT_DIR / "interview_dataset.json"
STATS_PATH  = OUTPUT_DIR / "dataset_stats.json"
PROGRESS_PATH = OUTPUT_DIR / "_progress.json"   # resume support

# ── ANSI stripping ─────────────────────────────────────────────────────────────
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b[@-_][0-9;]*|\r")

def _strip(text: str) -> str:
    return _ANSI.sub("", text).strip()

# ── Ollama call ────────────────────────────────────────────────────────────────

def call_ollama(prompt: str, timeout: int = 90) -> str:
    try:
        result = subprocess.run(
            ["ollama", "run", "--nowordwrap", MODEL],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            err = _strip(result.stderr.decode("utf-8", errors="replace"))
            return f"ERROR: {err}"
        return _strip(result.stdout.decode("utf-8", errors="replace"))
    except subprocess.TimeoutExpired:
        return "ERROR: timeout"
    except FileNotFoundError:
        return "ERROR: ollama not found — run `ollama serve` first"
    except Exception as e:
        return f"ERROR: {e}"

# ── Prompt templates ───────────────────────────────────────────────────────────

def make_question_prompt(role: str, interview_type: str, difficulty: str,
                          existing_questions: list[str]) -> str:
    avoid = ""
    if existing_questions:
        last_5 = existing_questions[-5:]
        avoid = "\n\nDo NOT repeat any of these already-used questions:\n" + \
                "\n".join(f"- {q}" for q in last_5)

    type_instruction = {
        "HR": "a behavioral or situational question (cultural fit, teamwork, conflict, leadership)",
        "Technical": f"a technical question specific to the {role} role",
    }.get(interview_type, "a relevant interview question")

    difficulty_guide = {
        "easy":   "entry-level, straightforward, no deep expertise needed",
        "medium": "mid-level, requires real work experience",
        "hard":   "senior-level, requires deep expertise and nuanced thinking",
    }.get(difficulty, "moderately challenging")

    return f"""You are an expert interviewer with 15 years of experience.

Generate ONE {interview_type} interview question for a {role} candidate.
Difficulty: {difficulty} ({difficulty_guide}).
Type: {type_instruction}.

Rules:
- Output ONLY the question text
- No numbering, no labels, no preamble
- Be concise and professional
- Make it realistic and commonly asked{avoid}"""


def make_answer_prompt(role: str, interview_type: str, difficulty: str,
                        question: str) -> str:
    return f"""You are a senior {role} with 10+ years of experience giving a perfect interview answer.

QUESTION: {question}

Write an IDEAL answer for this {difficulty}-level {interview_type} interview question.

Rules:
- For HR questions: use the STAR method (Situation, Task, Action, Result)
- For Technical questions: be precise, include examples, mention trade-offs
- Length: 3–6 sentences
- Output ONLY the answer — no labels, no preamble, no "Answer:" prefix"""


# ── Parsing helpers ────────────────────────────────────────────────────────────

def clean_question(text: str) -> str:
    """Strip common LLM prefixes from question output."""
    for prefix in ["Question:", "Q:", "1.", "2.", "-", "*", "•", "Here's", "Here is"]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    # Remove surrounding quotes
    text = text.strip('"\'')
    return text.strip()


def clean_answer(text: str) -> str:
    """Strip common LLM prefixes from answer output."""
    for prefix in ["Answer:", "A:", "IDEAL ANSWER:", "Here's an ideal answer",
                   "Here is an ideal answer", "Ideal answer:"]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    return text.strip()


def is_valid_entry(entry: dict) -> bool:
    q = entry.get("question", "")
    a = entry.get("ideal_answer", "")
    return (
        len(q) >= 20 and
        len(a) >= 40 and
        not q.startswith("ERROR") and
        not a.startswith("ERROR") and
        "?" in q  # must actually be a question
    )


# ── Progress (resume support) ──────────────────────────────────────────────────

def load_progress() -> list[dict]:
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH) as f:
            return json.load(f)
    return []


def save_progress(data: list[dict]):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_PATH, "w") as f:
        json.dump(data, f, indent=2)


# ── Statistics ─────────────────────────────────────────────────────────────────

def compute_stats(dataset: list[dict]) -> dict:
    from collections import Counter

    roles      = Counter(d["role"] for d in dataset)
    types      = Counter(d["type"] for d in dataset)
    diffs      = Counter(d["difficulty"] for d in dataset)
    q_lengths  = [len(d["question"].split()) for d in dataset]
    a_lengths  = [len(d["ideal_answer"].split()) for d in dataset]

    return {
        "total_samples": len(dataset),
        "by_role": dict(roles),
        "by_type": dict(types),
        "by_difficulty": dict(diffs),
        "avg_question_words": round(sum(q_lengths) / len(q_lengths), 1) if q_lengths else 0,
        "avg_answer_words":   round(sum(a_lengths) / len(a_lengths), 1) if a_lengths else 0,
        "min_answer_words":   min(a_lengths) if a_lengths else 0,
        "max_answer_words":   max(a_lengths) if a_lengths else 0,
        "generated_at": datetime.now().isoformat(),
        "model_used": MODEL,
    }


# ── Main generator ─────────────────────────────────────────────────────────────

def generate_dataset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Resume from progress if available
    dataset = load_progress()
    existing_keys = {
        (d["role"], d["type"], d["difficulty"], d["question"])
        for d in dataset
    }
    existing_questions_by_combo: dict[tuple, list[str]] = {}
    for d in dataset:
        key = (d["role"], d["type"], d["difficulty"])
        existing_questions_by_combo.setdefault(key, []).append(d["question"])

    # Build work list
    combos = [
        (role, itype, diff)
        for role  in ROLES
        for itype in INTERVIEW_TYPES
        for diff  in DIFFICULTIES
    ]

    total_target = len(combos) * SAMPLES_PER_COMBO
    generated    = len(dataset)
    skipped      = 0
    errors       = 0
    start_time   = time.time()

    print(f"\n{'='*60}")
    print(f"  AI Interview Dataset Generator")
    print(f"  Model  : {MODEL}")
    print(f"  Target : {total_target} samples")
    print(f"  Resumed: {generated} already done")
    print(f"{'='*60}\n")

    for combo_idx, (role, itype, diff) in enumerate(combos):
        combo_key = (role, itype, diff)
        done_for_combo = len(existing_questions_by_combo.get(combo_key, []))

        if done_for_combo >= SAMPLES_PER_COMBO:
            continue  # already have enough for this combo

        needed = SAMPLES_PER_COMBO - done_for_combo
        prev_questions = existing_questions_by_combo.get(combo_key, [])

        for sample_idx in range(needed):
            overall_idx = generated + 1
            pct = overall_idx / total_target * 100

            # ETA
            elapsed = time.time() - start_time
            rate = generated / elapsed if elapsed > 0 and generated > 0 else 0.1
            remaining = total_target - overall_idx
            eta_sec = int(remaining / rate) if rate > 0 else 0
            eta_str = f"{eta_sec // 60}m {eta_sec % 60}s"

            print(f"[{overall_idx:>3}/{total_target}] {pct:4.0f}% │ "
                  f"{role[:22]:<22} │ {itype:<9} │ {diff:<6} │ ETA {eta_str}")

            # ── Generate question ──────────────────────────────────────────
            q_prompt = make_question_prompt(role, itype, diff, prev_questions)
            raw_q    = call_ollama(q_prompt)
            question = clean_question(raw_q)

            if not question or question.startswith("ERROR") or len(question) < 20:
                print(f"         ⚠ Question generation failed — skipping")
                errors += 1
                continue

            if question in [d["question"] for d in dataset]:
                print(f"         ⚠ Duplicate question — skipping")
                skipped += 1
                continue

            # ── Generate ideal answer ──────────────────────────────────────
            a_prompt = make_answer_prompt(role, itype, diff, question)
            raw_a    = call_ollama(a_prompt)
            answer   = clean_answer(raw_a)

            if not answer or answer.startswith("ERROR") or len(answer) < 40:
                print(f"         ⚠ Answer generation failed — skipping")
                errors += 1
                continue

            entry = {
                "id":           f"{role.lower().replace(' ', '_')}_{itype.lower()}_{diff}_{len(dataset)+1:04d}",
                "role":         role,
                "type":         itype,
                "difficulty":   diff,
                "question":     question,
                "ideal_answer": answer,
                "generated_by": MODEL,
                "timestamp":    datetime.now().isoformat(),
            }

            if is_valid_entry(entry):
                dataset.append(entry)
                prev_questions.append(question)
                existing_questions_by_combo[combo_key] = prev_questions
                generated += 1
                print(f"         ✓ Q: {question[:70]}...")
            else:
                print(f"         ✗ Invalid entry (too short/missing '?') — skipping")
                skipped += 1

            # Save progress after every entry (resume support)
            save_progress(dataset)

    # ── Final save ─────────────────────────────────────────────────────────────
    with open(OUTPUT_PATH, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    stats = compute_stats(dataset)
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    # Cleanup progress file
    PROGRESS_PATH.unlink(missing_ok=True)

    elapsed_total = int(time.time() - start_time)
    print(f"\n{'='*60}")
    print(f"  ✅ Dataset generation complete!")
    print(f"  Samples  : {len(dataset)}")
    print(f"  Skipped  : {skipped}")
    print(f"  Errors   : {errors}")
    print(f"  Time     : {elapsed_total // 60}m {elapsed_total % 60}s")
    print(f"  Saved to : {OUTPUT_PATH}")
    print(f"  Stats    : {STATS_PATH}")
    print(f"\n  Stats preview:")
    for k, v in stats.items():
        if k not in ("generated_at", "model_used", "by_role"):
            print(f"    {k:<28}: {v}")
    print(f"{'='*60}\n")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    generate_dataset()