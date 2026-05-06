"""
Voice module - TTS integration for spoken coaching feedback.
Supports ElevenLabs (paid) or macOS say command (free).
"""
import os
import subprocess

# ElevenLabs pre-made voice IDs
VOICES = {
    "rachel": "21m00Tcm4TlvDq8ikWAM",
    "josh": "TxGEqnHWrfWFTfGW9XjX",
    "bella": "EXAVITQu4vr4xnSDxMaL",
    "adam": "pNInz6obpgDQGcFmaJgB",
    "sam": "yoZ06aMxZJJ28mfd3POQ",
}

# macOS say voices (free)
MACOS_VOICES = ["samantha", "alex", "victoria", "karen", "daniel"]

class VoiceCoach:
    """Text-to-speech coach using ElevenLabs API or macOS say."""

    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str = "samantha",
        use_elevenlabs: bool = False,
    ):
        """
        Initialize the voice coach.

        Args:
            api_key: ElevenLabs API key. Falls back to ELEVENLABS_API_KEY env var.
            voice_id: Voice name to use.
            use_elevenlabs: If True, use ElevenLabs (requires paid plan).
                           If False, use macOS say command (free).
        """
        self.use_elevenlabs = use_elevenlabs
        self.voice_id = voice_id

        if use_elevenlabs:
            self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "ElevenLabs API key required. Set ELEVENLABS_API_KEY env var or pass api_key."
                )
            # Resolve voice name to ID if needed
            self.voice_id = VOICES.get(voice_id.lower(), voice_id)

            from elevenlabs.client import ElevenLabs
            self.client = ElevenLabs(api_key=self.api_key)
        else:
            # Using macOS say command
            self.client = None

    def speak(self, text: str) -> None:
        """
        Convert text to speech and play it.

        Args:
            text: The text to speak aloud.
        """
        if self.use_elevenlabs:
            self._speak_elevenlabs(text)
        else:
            self._speak_macos(text)

    def _speak_elevenlabs(self, text: str) -> None:
        """Speak using ElevenLabs API."""
        try:
            from elevenlabs.play import play

            audio = self.client.text_to_speech.convert(
                text=text,
                voice_id=self.voice_id,
                model_id="eleven_turbo_v2_5",
            )
            play(audio)
        except Exception as e:
            print(f"Voice Coach error: {e}")

    def _speak_macos(self, text: str) -> None:
        """Speak using macOS say command (free, no API needed)."""
        try:
            # Escape quotes in text
            safe_text = text.replace('"', '\\"')
            cmd = ["say", "-v", self.voice_id, safe_text]
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"Voice Coach error: {e}")
