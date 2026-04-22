"""
app.py — AI Interview Simulator
Main GUI application using tkinter.
Orchestrates the full interview flow with voice I/O and multi-agent evaluation.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import json
import os
from datetime import datetime

# ── Agents & Utilities ─────────────────────────────────────────────────────────
from agents.interviewer_agent import generate_question
from agents.evaluator_agent import evaluate_answer
from agents.feedback_agent import generate_feedback
from agents.report_agent import generate_final_summary
from agents.session import InterviewSession
from utils.voice import speak_text, record_voice_answer

# ── Constants ──────────────────────────────────────────────────────────────────
MIN_QUESTIONS = 8
MAX_QUESTIONS = 10

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

INTERVIEW_TYPES = ["HR", "Technical", "Mixed"]

# ── Color Palette ──────────────────────────────────────────────────────────────
C = {
    "bg_dark":     "#0d1117",
    "bg_panel":    "#161b22",
    "bg_card":     "#1c2128",
    "accent":      "#58a6ff",
    "accent2":     "#3fb950",
    "accent3":     "#f78166",
    "accent4":     "#d2a679",
    "text":        "#e6edf3",
    "text_muted":  "#8b949e",
    "border":      "#30363d",
    "highlight":   "#1f6feb",
    "warn":        "#d29922",
}


# ══════════════════════════════════════════════════════════════════════════════
class AIInterviewApp:
    """Main application class managing all GUI frames and interview state."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AI Interview Simulator")
        self.root.geometry("1000x700")
        self.root.minsize(900, 620)
        self.root.configure(bg=C["bg_dark"])

        self.session: InterviewSession | None = None
        self.interview_running = False
        self.voice_enabled = tk.BooleanVar(value=True)
        self._interview_thread: threading.Thread | None = None

        self._build_styles()
        self._build_ui()
        self._show_frame("welcome")

    # ── Style Setup ───────────────────────────────────────────────────────────

    def _build_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("TFrame", background=C["bg_dark"])
        style.configure("Card.TFrame", background=C["bg_panel"], relief="flat")
        style.configure(
            "TLabel",
            background=C["bg_dark"],
            foreground=C["text"],
            font=("Courier New", 11),
        )
        style.configure(
            "Title.TLabel",
            background=C["bg_dark"],
            foreground=C["accent"],
            font=("Courier New", 22, "bold"),
        )
        style.configure(
            "Sub.TLabel",
            background=C["bg_dark"],
            foreground=C["text_muted"],
            font=("Courier New", 11),
        )
        style.configure(
            "Card.TLabel",
            background=C["bg_panel"],
            foreground=C["text"],
            font=("Courier New", 11),
        )
        style.configure(
            "Accent.TButton",
            background=C["highlight"],
            foreground=C["text"],
            font=("Courier New", 11, "bold"),
            padding=(12, 6),
        )
        style.configure(
            "TCombobox",
            fieldbackground=C["bg_card"],
            background=C["bg_card"],
            foreground=C["text"],
            selectbackground=C["highlight"],
            font=("Courier New", 11),
        )
        style.configure(
            "TProgressbar",
            troughcolor=C["bg_card"],
            background=C["accent"],
        )
        style.map(
            "Accent.TButton",
            background=[("active", C["accent"]), ("disabled", C["border"])],
        )

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        """Build all frames."""
        self.frames: dict[str, tk.Frame] = {}
        container = tk.Frame(self.root, bg=C["bg_dark"])
        container.pack(fill="both", expand=True)

        for name, builder in [
            ("welcome", self._build_welcome),
            ("interview", self._build_interview),
            ("feedback", self._build_feedback),
            ("report", self._build_report),
        ]:
            frame = tk.Frame(container, bg=C["bg_dark"])
            frame.place(relx=0, rely=0, relwidth=1, relheight=1)
            self.frames[name] = frame
            builder(frame)

    def _show_frame(self, name: str):
        self.frames[name].tkraise()

    # ── Welcome / Setup Frame ─────────────────────────────────────────────────

    def _build_welcome(self, parent: tk.Frame):
        # Header
        header = tk.Frame(parent, bg=C["bg_dark"])
        header.pack(pady=(50, 0))

        tk.Label(
            header, text="◆ AI INTERVIEW SIMULATOR ◆",
            font=("Courier New", 24, "bold"),
            fg=C["accent"], bg=C["bg_dark"],
        ).pack()
        tk.Label(
            header,
            text="Multi-Agent Evaluation  ·  Adaptive Questioning  ·  Voice Interface",
            font=("Courier New", 10),
            fg=C["text_muted"], bg=C["bg_dark"],
        ).pack(pady=(4, 0))

        # Config card
        card = tk.Frame(parent, bg=C["bg_panel"], bd=1, relief="solid",
                        highlightbackground=C["border"])
        card.pack(padx=80, pady=30, fill="x")

        inner = tk.Frame(card, bg=C["bg_panel"])
        inner.pack(padx=30, pady=25, fill="x")

        # Role selector
        tk.Label(inner, text="Select Role", font=("Courier New", 11, "bold"),
                 fg=C["accent"], bg=C["bg_panel"]).grid(row=0, column=0, sticky="w", pady=(0, 4))
        self.role_var = tk.StringVar(value=ROLES[0])
        role_cb = ttk.Combobox(inner, textvariable=self.role_var, values=ROLES,
                               state="readonly", width=32, font=("Courier New", 11))
        role_cb.grid(row=1, column=0, sticky="w", padx=(0, 40))

        # Interview type
        tk.Label(inner, text="Interview Type", font=("Courier New", 11, "bold"),
                 fg=C["accent"], bg=C["bg_panel"]).grid(row=0, column=1, sticky="w", pady=(0, 4))
        self.type_var = tk.StringVar(value="Mixed")
        type_cb = ttk.Combobox(inner, textvariable=self.type_var, values=INTERVIEW_TYPES,
                               state="readonly", width=16, font=("Courier New", 11))
        type_cb.grid(row=1, column=1, sticky="w")

        # Options row
        opts = tk.Frame(card, bg=C["bg_panel"])
        opts.pack(padx=30, pady=(0, 20), fill="x")

        voice_cb = tk.Checkbutton(
            opts, text=" 🎤 Voice Input / 🔊 TTS Output (requires microphone)",
            variable=self.voice_enabled,
            font=("Courier New", 10),
            fg=C["text_muted"], bg=C["bg_panel"],
            selectcolor=C["bg_card"],
            activebackground=C["bg_panel"],
        )
        voice_cb.pack(side="left")

        # Info strips
        info = tk.Frame(parent, bg=C["bg_dark"])
        info.pack(fill="x", padx=80, pady=(0, 20))

        for icon, text in [
            ("🤖", f"Minimum {MIN_QUESTIONS} questions with adaptive difficulty"),
            ("📊", "4-dimensional scoring: Clarity · Relevance · Depth · Structure"),
            ("💬", "Real-time feedback with suggested improved answers"),
        ]:
            row = tk.Frame(info, bg=C["bg_dark"])
            row.pack(anchor="w", pady=2)
            tk.Label(row, text=icon, font=("Courier New", 11),
                     fg=C["accent2"], bg=C["bg_dark"]).pack(side="left")
            tk.Label(row, text=f"  {text}", font=("Courier New", 10),
                     fg=C["text_muted"], bg=C["bg_dark"]).pack(side="left")

        # Start button
        start_btn = tk.Button(
            parent, text="  ▶  START INTERVIEW  ",
            font=("Courier New", 13, "bold"),
            fg=C["bg_dark"], bg=C["accent2"],
            activebackground=C["accent"], activeforeground=C["bg_dark"],
            relief="flat", cursor="hand2", padx=20, pady=10,
            command=self._start_interview,
        )
        start_btn.pack(pady=10)

    # ── Interview Frame ───────────────────────────────────────────────────────

    def _build_interview(self, parent: tk.Frame):
        # Top bar with progress
        top = tk.Frame(parent, bg=C["bg_panel"], height=50)
        top.pack(fill="x")
        top.pack_propagate(False)

        top_inner = tk.Frame(top, bg=C["bg_panel"])
        top_inner.pack(fill="both", expand=True, padx=20, pady=8)

        self.progress_label = tk.Label(
            top_inner, text="Question 0 / 0",
            font=("Courier New", 10, "bold"), fg=C["accent"], bg=C["bg_panel"],
        )
        self.progress_label.pack(side="left")

        self.difficulty_label = tk.Label(
            top_inner, text="Difficulty: medium",
            font=("Courier New", 10), fg=C["accent4"], bg=C["bg_panel"],
        )
        self.difficulty_label.pack(side="left", padx=20)

        self.score_avg_label = tk.Label(
            top_inner, text="Avg Score: –",
            font=("Courier New", 10), fg=C["accent2"], bg=C["bg_panel"],
        )
        self.score_avg_label.pack(side="right")

        self.progress_bar = ttk.Progressbar(
            parent, orient="horizontal", mode="determinate",
            style="TProgressbar",
        )
        self.progress_bar.pack(fill="x", padx=0, pady=0)

        # Main content area
        content = tk.Frame(parent, bg=C["bg_dark"])
        content.pack(fill="both", expand=True, padx=20, pady=10)

        # Left column: question & answer
        left = tk.Frame(content, bg=C["bg_dark"])
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Question box
        q_header = tk.Frame(left, bg=C["bg_dark"])
        q_header.pack(fill="x", pady=(0, 5))
        tk.Label(q_header, text="❓ QUESTION", font=("Courier New", 10, "bold"),
                 fg=C["accent"], bg=C["bg_dark"]).pack(side="left")
        self.q_type_badge = tk.Label(q_header, text="",
                                     font=("Courier New", 9),
                                     fg=C["bg_dark"], bg=C["accent4"], padx=6)
        self.q_type_badge.pack(side="right")

        self.question_text = scrolledtext.ScrolledText(
            left, height=6, wrap="word",
            font=("Courier New", 12),
            bg=C["bg_panel"], fg=C["text"],
            insertbackground=C["text"],
            relief="flat", padx=15, pady=12,
            state="disabled",
        )
        self.question_text.pack(fill="x", pady=(0, 12))

        # Answer box
        tk.Label(left, text="🎤 YOUR ANSWER (transcribed)",
                 font=("Courier New", 10, "bold"),
                 fg=C["accent2"], bg=C["bg_dark"]).pack(anchor="w", pady=(0, 5))

        self.answer_text = scrolledtext.ScrolledText(
            left, height=7, wrap="word",
            font=("Courier New", 11),
            bg=C["bg_card"], fg=C["text"],
            insertbackground=C["text"],
            relief="flat", padx=15, pady=12,
        )
        self.answer_text.pack(fill="both", expand=True, pady=(0, 8))

        # Control buttons
        btn_row = tk.Frame(left, bg=C["bg_dark"])
        btn_row.pack(fill="x")

        self.record_btn = tk.Button(
            btn_row, text="🎤  Record Answer",
            font=("Courier New", 11, "bold"),
            fg=C["bg_dark"], bg=C["accent"],
            activebackground=C["highlight"],
            relief="flat", cursor="hand2", padx=14, pady=7,
            command=self._record_answer_manual,
        )
        self.record_btn.pack(side="left", padx=(0, 8))

        self.submit_btn = tk.Button(
            btn_row, text="✓  Submit Answer",
            font=("Courier New", 11, "bold"),
            fg=C["bg_dark"], bg=C["accent2"],
            activebackground=C["accent"],
            relief="flat", cursor="hand2", padx=14, pady=7,
            command=self._submit_typed_answer,
            state="disabled",
        )
        self.submit_btn.pack(side="left", padx=(0, 8))

        self.skip_btn = tk.Button(
            btn_row, text="⏭  Skip",
            font=("Courier New", 10),
            fg=C["text_muted"], bg=C["bg_card"],
            relief="flat", cursor="hand2", padx=10, pady=7,
            command=self._skip_question,
            state="disabled",
        )
        self.skip_btn.pack(side="left")

        # Right column: status + live scores
        right = tk.Frame(content, bg=C["bg_panel"], width=240)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        rinner = tk.Frame(right, bg=C["bg_panel"])
        rinner.pack(fill="both", expand=True, padx=15, pady=15)

        tk.Label(rinner, text="STATUS", font=("Courier New", 9, "bold"),
                 fg=C["text_muted"], bg=C["bg_panel"]).pack(anchor="w")

        self.status_label = tk.Label(
            rinner, text="Ready to start...",
            font=("Courier New", 10), fg=C["accent"],
            bg=C["bg_panel"], wraplength=210, justify="left",
        )
        self.status_label.pack(anchor="w", pady=(4, 16))

        tk.Label(rinner, text="LAST SCORES", font=("Courier New", 9, "bold"),
                 fg=C["text_muted"], bg=C["bg_panel"]).pack(anchor="w")

        self.score_labels: dict[str, tk.Label] = {}
        for dim in ["Clarity", "Relevance", "Depth", "Structure", "Overall"]:
            row = tk.Frame(rinner, bg=C["bg_panel"])
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f"{dim}:", font=("Courier New", 10),
                     fg=C["text_muted"], bg=C["bg_panel"], width=10, anchor="w").pack(side="left")
            lbl = tk.Label(row, text="–", font=("Courier New", 10, "bold"),
                           fg=C["accent"], bg=C["bg_panel"])
            lbl.pack(side="left")
            self.score_labels[dim.lower()] = lbl

        # Score history mini-bar area
        tk.Label(rinner, text="\nSCORE HISTORY",
                 font=("Courier New", 9, "bold"),
                 fg=C["text_muted"], bg=C["bg_panel"]).pack(anchor="w")

        self.history_frame = tk.Frame(rinner, bg=C["bg_panel"])
        self.history_frame.pack(fill="x", pady=(4, 0))

        # Bottom status bar
        self.bottom_status = tk.Label(
            parent, text="",
            font=("Courier New", 10), fg=C["text_muted"],
            bg=C["bg_card"], anchor="w", padx=15,
        )
        self.bottom_status.pack(fill="x", side="bottom", ipady=5)

    # ── Feedback Frame ────────────────────────────────────────────────────────

    def _build_feedback(self, parent: tk.Frame):
        tk.Label(parent, text="📋 ANSWER FEEDBACK",
                 font=("Courier New", 16, "bold"),
                 fg=C["accent"], bg=C["bg_dark"]).pack(pady=(20, 5))

        self.fb_q_label = tk.Label(parent, text="",
                                   font=("Courier New", 10, "italic"),
                                   fg=C["text_muted"], bg=C["bg_dark"],
                                   wraplength=860, justify="left")
        self.fb_q_label.pack(padx=40, anchor="w")

        # Scores row
        self.fb_scores_frame = tk.Frame(parent, bg=C["bg_dark"])
        self.fb_scores_frame.pack(fill="x", padx=40, pady=10)

        # Feedback sections
        sections = tk.Frame(parent, bg=C["bg_dark"])
        sections.pack(fill="both", expand=True, padx=40, pady=5)

        left_col = tk.Frame(sections, bg=C["bg_dark"])
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 8))
        right_col = tk.Frame(sections, bg=C["bg_dark"])
        right_col.pack(side="right", fill="both", expand=True, padx=(8, 0))

        self.fb_strengths = self._feedback_box(left_col, "✅ Strengths", C["accent2"])
        self.fb_weaknesses = self._feedback_box(left_col, "⚠️ Weaknesses", C["accent3"])
        self.fb_suggestions = self._feedback_box(right_col, "💡 Suggestions", C["warn"])
        self.fb_improved = self._feedback_box(right_col, "🌟 Improved Answer", C["accent"])

        # Continue button
        self.fb_continue_btn = tk.Button(
            parent, text="  ▶  Continue to Next Question  ",
            font=("Courier New", 12, "bold"),
            fg=C["bg_dark"], bg=C["accent2"],
            activebackground=C["accent"], activeforeground=C["bg_dark"],
            relief="flat", cursor="hand2", padx=16, pady=8,
            command=self._continue_interview,
        )
        self.fb_continue_btn.pack(pady=15)

    def _feedback_box(self, parent, title: str, color: str) -> scrolledtext.ScrolledText:
        tk.Label(parent, text=title, font=("Courier New", 10, "bold"),
                 fg=color, bg=C["bg_dark"]).pack(anchor="w", pady=(8, 3))
        box = scrolledtext.ScrolledText(
            parent, height=5, wrap="word",
            font=("Courier New", 10),
            bg=C["bg_panel"], fg=C["text"],
            relief="flat", padx=10, pady=8,
            state="disabled",
        )
        box.pack(fill="both", expand=True)
        return box

    # ── Report Frame ──────────────────────────────────────────────────────────

    def _build_report(self, parent: tk.Frame):
        tk.Label(parent, text="🏆 INTERVIEW COMPLETE — FINAL REPORT",
                 font=("Courier New", 16, "bold"),
                 fg=C["accent"], bg=C["bg_dark"]).pack(pady=(20, 5))

        # Top score cards row
        self.report_cards_frame = tk.Frame(parent, bg=C["bg_dark"])
        self.report_cards_frame.pack(fill="x", padx=40, pady=10)

        # Tier badge
        self.report_tier = tk.Label(parent, text="",
                                    font=("Courier New", 14, "bold"),
                                    fg=C["accent4"], bg=C["bg_dark"])
        self.report_tier.pack(pady=(0, 5))

        # AI Narrative summary
        tk.Label(parent, text="AI ASSESSMENT",
                 font=("Courier New", 9, "bold"),
                 fg=C["text_muted"], bg=C["bg_dark"]).pack(anchor="w", padx=40)

        self.report_summary = scrolledtext.ScrolledText(
            parent, height=5, wrap="word",
            font=("Courier New", 11),
            bg=C["bg_panel"], fg=C["text"],
            relief="flat", padx=14, pady=10,
            state="disabled",
        )
        self.report_summary.pack(fill="x", padx=40, pady=(4, 10))

        # Detail section (strengths / weaknesses / suggestions)
        detail_row = tk.Frame(parent, bg=C["bg_dark"])
        detail_row.pack(fill="both", expand=True, padx=40)

        self.rpt_strengths = self._report_detail_box(detail_row, "✅ Key Strengths", C["accent2"])
        self.rpt_weaknesses = self._report_detail_box(detail_row, "⚠️ Patterns to Improve", C["accent3"])
        self.rpt_suggestions = self._report_detail_box(detail_row, "💡 Top Suggestions", C["warn"])

        # Buttons
        btn_row = tk.Frame(parent, bg=C["bg_dark"])
        btn_row.pack(pady=15)

        tk.Button(
            btn_row, text="💾  Save Report (JSON)",
            font=("Courier New", 10),
            fg=C["text"], bg=C["bg_card"],
            relief="flat", cursor="hand2", padx=12, pady=6,
            command=self._save_report,
        ).pack(side="left", padx=6)

        tk.Button(
            btn_row, text="🔄  New Interview",
            font=("Courier New", 10, "bold"),
            fg=C["bg_dark"], bg=C["accent"],
            relief="flat", cursor="hand2", padx=12, pady=6,
            command=self._restart,
        ).pack(side="left", padx=6)

    def _report_detail_box(self, parent, title: str, color: str) -> scrolledtext.ScrolledText:
        col = tk.Frame(parent, bg=C["bg_dark"])
        col.pack(side="left", fill="both", expand=True, padx=(0, 6))
        tk.Label(col, text=title, font=("Courier New", 10, "bold"),
                 fg=color, bg=C["bg_dark"]).pack(anchor="w", pady=(0, 4))
        box = scrolledtext.ScrolledText(
            col, height=8, wrap="word",
            font=("Courier New", 10),
            bg=C["bg_panel"], fg=C["text"],
            relief="flat", padx=10, pady=8,
            state="disabled",
        )
        box.pack(fill="both", expand=True)
        return box

    # ── Interview Logic ───────────────────────────────────────────────────────

    def _start_interview(self):
        role = self.role_var.get()
        itype = self.type_var.get()

        self.session = InterviewSession(role=role, interview_type=itype)
        self._show_frame("interview")
        self._set_status(f"Starting {itype} interview for {role}...")

        self._update_top_bar()
        self.progress_bar["maximum"] = MIN_QUESTIONS
        self.progress_bar["value"] = 0

        # Run the first question in a background thread
        self._run_next_question()

    def _run_next_question(self):
        """Launch the next question generation in a background thread."""
        self.interview_running = True
        self._set_buttons_enabled(False)
        self._set_status("⏳ Generating question...")

        thread = threading.Thread(target=self._question_thread, daemon=True)
        thread.start()

    def _question_thread(self):
        """Background thread: generate question, display, speak, then listen."""
        topics = self.session.questions_asked[:]
        q_num = self.session.question_number + 1
        difficulty = self.session.current_difficulty

        question = generate_question(
            role=self.session.role,
            difficulty=difficulty,
            interview_type=self.session.interview_type,
            previous_topics=topics,
            question_number=q_num,
        )

        # Display question in GUI (thread-safe)
        self.root.after(0, self._display_question, question, q_num, difficulty)

        # Speak question via TTS if enabled
        if self.voice_enabled.get():
            speak_text(question, status_callback=lambda s: self.root.after(0, self._set_status, s))

        # Auto-record if voice enabled
        if self.voice_enabled.get():
            self.root.after(0, self._start_recording_flow, question)
        else:
            # Manual text mode — enable submit button
            self.root.after(0, self._enable_manual_input)

    def _display_question(self, question: str, q_num: int, difficulty: str):
        """Update the question area in the GUI."""
        self._set_text_widget(self.question_text, question)
        self.q_type_badge.config(text=f" {difficulty.upper()} ")

        # Clear previous answer
        self.answer_text.delete("1.0", "end")
        self._set_status(f"Question {q_num} ready.")

        self._update_top_bar()

    def _start_recording_flow(self, question: str):
        """Begin voice recording in a thread."""
        self._set_status("🎤 Recording... speak your answer clearly.")
        self._set_buttons_enabled(False)

        def record():
            answer = record_voice_answer(
                timeout=15,
                phrase_limit=90,
                status_callback=lambda s: self.root.after(0, self._set_status, s),
            )
            self.root.after(0, self._on_answer_received, question, answer)

        threading.Thread(target=record, daemon=True).start()

    def _record_answer_manual(self):
        """Manually trigger voice recording (for manual-mode or re-record)."""
        question = self.question_text.get("1.0", "end").strip()
        if not question:
            return
        self._set_buttons_enabled(False)
        self._start_recording_flow(question)

    def _enable_manual_input(self):
        """Enable text typing mode when voice is off."""
        self.answer_text.config(state="normal")
        self.submit_btn.config(state="normal")
        self.skip_btn.config(state="normal")
        self._set_status("✍️ Type your answer and click Submit, or Skip.")

    def _submit_typed_answer(self):
        """Submit manually typed answer."""
        question = self.question_text.get("1.0", "end").strip()
        answer = self.answer_text.get("1.0", "end").strip()
        self._on_answer_received(question, answer)

    def _skip_question(self):
        """Skip current question with empty answer."""
        question = self.question_text.get("1.0", "end").strip()
        self._on_answer_received(question, "")

    def _on_answer_received(self, question: str, answer: str):
        """Called when answer is ready (from voice or text)."""
        # Display transcribed answer
        if answer:
            self._set_text_widget(self.answer_text, answer)
        else:
            self._set_text_widget(self.answer_text, "[No answer provided]")

        self._set_status("⚙️ Evaluating answer...")
        self._set_buttons_enabled(False)

        # Evaluate in background
        def evaluate():
            evaluation = evaluate_answer(question, answer)
            feedback = generate_feedback(question, answer, evaluation)
            self.root.after(0, self._on_evaluation_done, question, answer, evaluation, feedback)

        threading.Thread(target=evaluate, daemon=True).start()

    def _on_evaluation_done(self, question, answer, evaluation, feedback):
        """Called when evaluation is complete — update scores and show feedback."""
        # Record in session
        self.session.add_record(question, answer, evaluation, feedback)

        # Update live score display
        self._update_score_display(evaluation)
        self._update_top_bar()
        self._add_score_history_bar(evaluation["overall"])

        # Show feedback frame
        self._populate_feedback(question, answer, evaluation, feedback)
        self._show_frame("feedback")

    def _continue_interview(self):
        """User clicks 'Continue' on feedback screen."""
        q_count = self.session.question_number

        if q_count >= MAX_QUESTIONS:
            self._finish_interview()
        elif q_count >= MIN_QUESTIONS:
            # Optional: could ask if user wants to continue, for now auto-end at MAX
            self._finish_interview()
        else:
            self._show_frame("interview")
            self._run_next_question()

    def _finish_interview(self):
        """End the interview and show the final report."""
        self._set_status("🏁 Interview complete! Generating report...")

        def build_report():
            report = self.session.build_final_report()
            summary = generate_final_summary(report)
            self.root.after(0, self._display_report, report, summary)

        threading.Thread(target=build_report, daemon=True).start()

    # ── Feedback Population ───────────────────────────────────────────────────

    def _populate_feedback(self, question, answer, evaluation, feedback):
        """Fill the feedback frame with current Q&A evaluation data."""
        self.fb_q_label.config(text=f"Q: {question}")

        # Score badges
        for widget in self.fb_scores_frame.winfo_children():
            widget.destroy()

        score_items = [
            ("Clarity", evaluation["clarity"], C["accent"]),
            ("Relevance", evaluation["relevance"], C["accent2"]),
            ("Depth", evaluation["depth"], C["warn"]),
            ("Structure", evaluation["structure"], C["accent4"]),
            ("Overall", evaluation["overall"], C["accent3"]),
        ]
        for label, score, color in score_items:
            box = tk.Frame(self.fb_scores_frame, bg=C["bg_panel"], padx=14, pady=8)
            box.pack(side="left", padx=5)
            tk.Label(box, text=f"{score}", font=("Courier New", 18, "bold"),
                     fg=color, bg=C["bg_panel"]).pack()
            tk.Label(box, text=label, font=("Courier New", 8),
                     fg=C["text_muted"], bg=C["bg_panel"]).pack()

        # Explanation
        explanation = evaluation.get("explanation", "")
        tk.Label(self.fb_scores_frame,
                 text=explanation,
                 font=("Courier New", 9, "italic"),
                 fg=C["text_muted"], bg=C["bg_dark"],
                 wraplength=400, justify="left").pack(side="left", padx=15)

        # Text boxes
        self._set_text_widget(self.fb_strengths, feedback.get("strengths", ""))
        self._set_text_widget(self.fb_weaknesses, feedback.get("weaknesses", ""))
        self._set_text_widget(self.fb_suggestions, feedback.get("suggestions", ""))
        self._set_text_widget(self.fb_improved, feedback.get("improved_answer", ""))

        # Update continue button label
        q_count = self.session.question_number
        if q_count >= MIN_QUESTIONS:
            self.fb_continue_btn.config(text="  🏁  View Final Report  ")
        else:
            remaining = MIN_QUESTIONS - q_count
            self.fb_continue_btn.config(
                text=f"  ▶  Next Question  ({remaining} more to go)  "
                if remaining > 0 else "  🏁  View Final Report  "
            )

    # ── Report Display ────────────────────────────────────────────────────────

    def _display_report(self, report: dict, summary: str):
        avg = report["average_scores"]

        # Score cards
        for widget in self.report_cards_frame.winfo_children():
            widget.destroy()

        score_items = [
            ("Clarity", avg["clarity"], C["accent"]),
            ("Relevance", avg["relevance"], C["accent2"]),
            ("Depth", avg["depth"], C["warn"]),
            ("Structure", avg["structure"], C["accent4"]),
            ("Overall", avg["overall"], C["accent3"]),
        ]
        for label, score, color in score_items:
            card = tk.Frame(self.report_cards_frame, bg=C["bg_panel"], padx=20, pady=12)
            card.pack(side="left", padx=6, fill="y")
            tk.Label(card, text=f"{score}", font=("Courier New", 22, "bold"),
                     fg=color, bg=C["bg_panel"]).pack()
            tk.Label(card, text=label, font=("Courier New", 9),
                     fg=C["text_muted"], bg=C["bg_panel"]).pack()

        # Tier
        self.report_tier.config(text=report["performance_tier"] + " — " + report["tier_message"])

        # AI Summary
        self._set_text_widget(self.report_summary, summary)

        # Aggregate strengths / weaknesses / suggestions
        strengths_text = "\n\n".join(
            f"Q{i+1}: {s}" for i, s in enumerate(report["overall_strengths"]) if s
        ) or "N/A"
        weaknesses_text = "\n\n".join(
            f"Q{i+1}: {w}" for i, w in enumerate(report["overall_weaknesses"]) if w
        ) or "N/A"
        suggestions_text = "\n\n".join(
            f"Q{i+1}: {s}" for i, s in enumerate(report["overall_suggestions"]) if s
        ) or "N/A"

        self._set_text_widget(self.rpt_strengths, strengths_text)
        self._set_text_widget(self.rpt_weaknesses, weaknesses_text)
        self._set_text_widget(self.rpt_suggestions, suggestions_text)

        self._show_frame("report")

    # ── Helper Methods ────────────────────────────────────────────────────────

    def _set_status(self, msg: str):
        self.status_label.config(text=msg)
        self.bottom_status.config(text=f"  {msg}")

    def _set_text_widget(self, widget: scrolledtext.ScrolledText, text: str):
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", text)
        widget.config(state="disabled")

    def _set_buttons_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        self.record_btn.config(state=state)
        self.submit_btn.config(state=state if not self.voice_enabled.get() else "disabled")
        self.skip_btn.config(state=state)

    def _update_top_bar(self):
        q_count = self.session.question_number if self.session else 0
        self.progress_label.config(text=f"Question {q_count} / {MIN_QUESTIONS}+")
        diff = self.session.current_difficulty if self.session else "medium"
        self.difficulty_label.config(text=f"Difficulty: {diff}")
        self.progress_bar["value"] = min(q_count, MIN_QUESTIONS)

        if self.session and self.session.qa_records:
            avg = self.session.get_average_scores()["overall"]
            self.score_avg_label.config(text=f"Avg Score: {avg}/10")

    def _update_score_display(self, evaluation: dict):
        colors = {"clarity": C["accent"], "relevance": C["accent2"],
                  "depth": C["warn"], "structure": C["accent4"], "overall": C["accent3"]}
        for dim, lbl in self.score_labels.items():
            val = evaluation.get(dim, "–")
            color = colors.get(dim, C["text"])
            lbl.config(text=str(val), fg=color)

    def _add_score_history_bar(self, score: float):
        """Add a tiny color-coded score bar to the history panel."""
        height = int((score / 10) * 40)
        color = C["accent2"] if score >= 7 else C["warn"] if score >= 4 else C["accent3"]

        bar = tk.Frame(self.history_frame, bg=color, width=12, height=max(4, height))
        bar.pack(side="left", padx=1, anchor="s")

        score_lbl = tk.Label(self.history_frame, text=str(int(score)),
                             font=("Courier New", 7),
                             fg=C["text_muted"], bg=C["bg_panel"])
        score_lbl.pack(side="left", anchor="s")

    def _save_report(self):
        if not self.session:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"interview_report_{timestamp}.json"
        self.session.save_to_file(filename)
        messagebox.showinfo("Saved", f"Report saved to:\n{os.path.abspath(filename)}")

    def _restart(self):
        self.session = None
        self.interview_running = False

        # Clear score history bars
        for w in self.history_frame.winfo_children():
            w.destroy()

        # Reset score labels
        for lbl in self.score_labels.values():
            lbl.config(text="–")

        self._show_frame("welcome")


# ══════════════════════════════════════════════════════════════════════════════

def main():
    root = tk.Tk()
    app = AIInterviewApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
