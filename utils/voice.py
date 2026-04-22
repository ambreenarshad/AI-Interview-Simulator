"""
voice.py — Text-to-Speech and Speech-to-Text utilities
TTS: gTTS (Google TTS) with pygame for audio playback
STT: SpeechRecognition with Google Web Speech API (local microphone)
Falls back gracefully if audio hardware is unavailable.

Windows fixes applied:
  - pygame.mixer.music.unload() before os.unlink() to release file lock (WinError 32)
  - pyttsx3 used as primary TTS fallback (works fully offline on Windows)
"""

import os
import tempfile
import speech_recognition as sr

# Try importing gTTS and pygame for TTS
try:
    from gtts import gTTS
    import pygame
    pygame.mixer.init()
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

# Try importing pyttsx3 as fallback TTS (works great on Windows, fully offline)
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except Exception:
    PYTTSX3_AVAILABLE = False


# ─── Text-to-Speech ────────────────────────────────────────────────────────────

def speak_text(text: str, status_callback=None):
    """
    Convert text to speech and play it.
    On Windows: tries pyttsx3 first (offline, no file-lock issues),
    then gTTS+pygame, then prints.
    """
    if status_callback:
        status_callback("🔊 Speaking question...")

    # On Windows pyttsx3 is more reliable — try it first
    if PYTTSX3_AVAILABLE:
        _speak_pyttsx3(text)
    elif TTS_AVAILABLE:
        _speak_gtts(text)
    else:
        print(f"[TTS unavailable] Would say: {text}")

    if status_callback:
        status_callback("✅ Done speaking.")


def _speak_gtts(text: str):
    """Use gTTS + pygame to play speech."""
    tmp_path = None
    try:
        tts = gTTS(text=text, lang="en", slow=False)
        # Use delete=False so we control deletion timing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            tmp_path = f.name
        tts.save(tmp_path)

        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        # WINDOWS FIX: must explicitly unload before deleting
        # pygame holds an open file handle until unload() is called
        pygame.mixer.music.unload()

    except Exception as e:
        print(f"[gTTS Error] {e}")
        if PYTTSX3_AVAILABLE:
            _speak_pyttsx3(text)
    finally:
        # Clean up temp file — silently ignore if still locked
        if tmp_path and os.path.exists(tmp_path):
            try:
                pygame.mixer.music.unload()
                os.unlink(tmp_path)
            except Exception:
                pass  # OS will clean up temp files eventually


def _speak_pyttsx3(text: str):
    """Use pyttsx3 offline TTS (best option on Windows — no file locks, no internet)."""
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)   # slightly slower = clearer
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"[pyttsx3 Error] {e}")


# ─── Speech-to-Text ────────────────────────────────────────────────────────────

recognizer = sr.Recognizer()
recognizer.energy_threshold = 300
recognizer.pause_threshold = 1.5   # seconds of silence to end phrase
recognizer.dynamic_energy_threshold = True


def record_voice_answer(
    timeout: int = 15,
    phrase_limit: int = 90,
    status_callback=None,
) -> str:
    """
    Record voice from microphone and transcribe via Google Web Speech.
    Returns transcribed text, or empty string on failure.

    timeout: seconds to wait for speech to start
    phrase_limit: max seconds to record a single answer
    status_callback: optional function(str) to push status updates to GUI
    """
    if status_callback:
        status_callback("🎤 Listening... speak your answer now.")

    try:
        with sr.Microphone() as source:
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
        msg = "[No speech detected — timed out]"
        if status_callback:
            status_callback(msg)
        return ""
    except sr.UnknownValueError:
        msg = "[Could not understand audio — please speak clearly]"
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
