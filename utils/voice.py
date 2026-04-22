"""
voice.py — Text-to-Speech and Speech-to-Text utilities
TTS: gTTS (Google TTS, offline-like via local playback) with pygame for audio
STT: SpeechRecognition with Google Web Speech API (local microphone)
Falls back gracefully if audio hardware is unavailable.
"""

import os
import tempfile
import threading
import speech_recognition as sr

# Try importing gTTS and pygame for TTS
try:
    from gtts import gTTS
    import pygame
    pygame.mixer.init()
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

# Try importing pyttsx3 as fallback TTS
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except Exception:
    PYTTSX3_AVAILABLE = False


# ─── Text-to-Speech ────────────────────────────────────────────────────────────

def speak_text(text: str, status_callback=None):
    """
    Convert text to speech and play it.
    Tries gTTS + pygame first, falls back to pyttsx3, then prints only.
    status_callback: optional function(str) to update GUI status label.
    """
    if status_callback:
        status_callback("🔊 Speaking question...")

    if TTS_AVAILABLE:
        _speak_gtts(text)
    elif PYTTSX3_AVAILABLE:
        _speak_pyttsx3(text)
    else:
        print(f"[TTS unavailable] Would say: {text}")

    if status_callback:
        status_callback("✅ Done speaking.")


def _speak_gtts(text: str):
    """Use gTTS + pygame to play speech."""
    try:
        tts = gTTS(text=text, lang="en", slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            tmp_path = f.name
        tts.save(tmp_path)

        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        os.unlink(tmp_path)
    except Exception as e:
        print(f"[gTTS Error] {e}")
        if PYTTSX3_AVAILABLE:
            _speak_pyttsx3(text)


def _speak_pyttsx3(text: str):
    """Use pyttsx3 offline TTS as fallback."""
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"[pyttsx3 Error] {e}")


# ─── Speech-to-Text ────────────────────────────────────────────────────────────

recognizer = sr.Recognizer()
recognizer.energy_threshold = 300
recognizer.pause_threshold = 1.5  # seconds of silence to end phrase
recognizer.dynamic_energy_threshold = True


def record_voice_answer(
    timeout: int = 15,
    phrase_limit: int = 60,
    status_callback=None,
) -> str:
    """
    Record voice from microphone and transcribe via Google Web Speech.
    Returns transcribed text or an error/empty string.

    timeout: seconds to wait for speech to start
    phrase_limit: max seconds to record a single answer
    status_callback: optional function(str) to push status updates to GUI
    """
    if status_callback:
        status_callback("🎤 Listening... speak your answer now.")

    try:
        with sr.Microphone() as source:
            # Brief ambient noise calibration
            recognizer.adjust_for_ambient_noise(source, duration=0.5)

            audio = recognizer.listen(
                source,
                timeout=timeout,
                phrase_time_limit=phrase_limit,
            )

        if status_callback:
            status_callback("⏳ Transcribing your answer...")

        text = recognizer.recognize_google(audio)
        return text.strip()

    except sr.WaitTimeoutError:
        msg = "[No speech detected within timeout]"
        if status_callback:
            status_callback(msg)
        return ""
    except sr.UnknownValueError:
        msg = "[Could not understand audio]"
        if status_callback:
            status_callback(msg)
        return ""
    except sr.RequestError as e:
        msg = f"[STT service error: {e}]"
        if status_callback:
            status_callback(msg)
        return ""
    except OSError as e:
        msg = f"[Microphone error: {e}]"
        if status_callback:
            status_callback(msg)
        return ""
    except Exception as e:
        msg = f"[Unexpected STT error: {e}]"
        if status_callback:
            status_callback(msg)
        return ""
