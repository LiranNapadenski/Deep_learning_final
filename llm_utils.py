import os
import random
import threading
import time

import google.generativeai as genai


_lock = threading.Lock()
_last_call_monotonic = 0.0


def _min_interval_seconds() -> float:
    """
    Global pacing for *all* Gemini calls in this process.

    Configure via either:
      - GEMINI_MIN_INTERVAL_SECONDS (highest priority), or
      - GEMINI_RPM (requests per minute)
    """
    v = os.environ.get("GEMINI_MIN_INTERVAL_SECONDS")
    if v:
        try:
            return max(0.0, float(v))
        except ValueError:
            pass

    rpm = os.environ.get("GEMINI_RPM")
    if rpm:
        try:
            rpm_f = float(rpm)
            if rpm_f > 0:
                return 60.0 / rpm_f
        except ValueError:
            pass

    # Safe-ish default for shared/free tiers.
    return 5.0


def _pace() -> None:
    global _last_call_monotonic
    interval = _min_interval_seconds()
    if interval <= 0:
        return

    with _lock:
        now = time.monotonic()
        wait = (_last_call_monotonic + interval) - now
        if wait > 0:
            time.sleep(wait)
        _last_call_monotonic = time.monotonic()


def gemini_generate(
    prompt: str,
    *,
    model_name: str = "gemini-3.1-flash-lite-preview",
    temperature: float = 0.0,
    max_retries: int = 12,
) -> str:
    """
    Single entry point for Gemini calls.
    - Enforces process-wide RPM pacing (see _min_interval_seconds()).
    - Retries on quota/rate-limit errors with exponential backoff.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    for attempt in range(max_retries):
        _pace()
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=temperature),
            )
            return (response.text or "").strip()
        except Exception as e:
            msg = str(e)
            is_rate_limited = (
                "429" in msg
                or "503" in msg
                or "quota" in msg.lower()
                or "resourceexhausted" in msg.lower()
                or "rate" in msg.lower()
                or "unavailable" in msg.lower()
            )
            if not is_rate_limited:
                raise

            # Exponential backoff with jitter; keeps us within RPM even if server-side
            # enforcement is bursty.
            sleep_s = min(120.0, (2.0**attempt) + random.uniform(0.0, 1.0))
            print(f"Gemini rate limited (attempt {attempt+1}/{max_retries}). Sleeping {sleep_s:.1f}s...")
            time.sleep(sleep_s)

    raise RuntimeError("Gemini rate limits exhausted; max retries reached.")

