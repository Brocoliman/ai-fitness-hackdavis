"""
Standalone webcam squat analyzer with AI voice coaching.
Run: python webcam_demo.py [--mode pro] [--provider claude]
Press 'q' to quit, 'r' to reset counters.
"""
import argparse
import os
import threading
import cv2
from analyzer import SquatAnalyzer


def create_feedback_callback(ai_coach, voice_coach):
    """Create a callback that processes AI feedback in a background thread."""

    def on_rep_complete(rep_data):
        def process_feedback():
            try:
                feedback = ai_coach.get_feedback(rep_data)
                print(f"[AI Coach] {feedback}")
                voice_coach.speak(feedback)
            except Exception as e:
                print(f"[AI Coach Error] {e}")

        thread = threading.Thread(target=process_feedback, daemon=True)
        thread.start()

    return on_rep_complete


def main():
    parser = argparse.ArgumentParser(description="AI Squat Trainer — Webcam")
    parser.add_argument("--mode", choices=["beginner", "pro"], default="beginner")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")

    # AI Coach arguments
    parser.add_argument(
        "--provider",
        choices=["claude", "openai"],
        default=None,
        help="AI provider for coaching feedback (claude or openai)",
    )
    parser.add_argument(
        "--anthropic-key",
        default=None,
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--openai-key",
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--elevenlabs-key",
        default=None,
        help="ElevenLabs API key (or set ELEVENLABS_API_KEY env var)",
    )
    parser.add_argument(
        "--voice-id",
        default="samantha",
        help="Voice name (macOS: samantha, alex, victoria; ElevenLabs: rachel, josh)",
    )
    parser.add_argument(
        "--use-elevenlabs",
        action="store_true",
        help="Use ElevenLabs TTS (requires paid plan). Default: macOS say (free)",
    )

    args = parser.parse_args()

    # Set up AI coaching if provider is specified
    on_rep_callback = None
    if args.provider:
        try:
            from ai_coach import AICoach
            from voice import VoiceCoach

            # Set API keys from CLI args if provided
            if args.anthropic_key:
                os.environ["ANTHROPIC_API_KEY"] = args.anthropic_key
            if args.openai_key:
                os.environ["OPENAI_API_KEY"] = args.openai_key
            if args.elevenlabs_key:
                os.environ["ELEVENLABS_API_KEY"] = args.elevenlabs_key

            ai_coach = AICoach(provider=args.provider)
            voice_coach = VoiceCoach(
                voice_id=args.voice_id,
                use_elevenlabs=args.use_elevenlabs,
            )
            on_rep_callback = create_feedback_callback(ai_coach, voice_coach)
            tts_name = "ElevenLabs" if args.use_elevenlabs else "macOS say"
            print(f"AI Coach enabled: {args.provider} + {tts_name} voice")
        except Exception as e:
            print(f"Warning: Could not initialize AI coach: {e}")
            print("Continuing without AI coaching...")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Error: cannot open camera")
        return

    analyzer = SquatAnalyzer(mode=args.mode, on_rep_complete=on_rep_callback)
    print(f"Mode: {args.mode} | Press 'q' to quit, 'r' to reset counters")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = analyzer.process_frame(frame)
        cv2.imshow("AI Squat Trainer", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            analyzer.reset()
            print("Counters reset.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Final — Correct: {analyzer.correct_count} | Incorrect: {analyzer.incorrect_count}")


if __name__ == "__main__":
    main()
