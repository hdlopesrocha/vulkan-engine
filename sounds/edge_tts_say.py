#!/usr/bin/env python3
"""Synthesize stdin to an MP3 file with Microsoft Edge neural TTS.

This is the synthesis half of the shared alert voice: sounds/say.sh (runtime
diagnostics, cloc count, commit subjects) and `make sounds` (checked-in phase
alerts) both call it, so every alert uses the same voice and endpoint fallback
as the opencode-talk project (scripts/edge_tts_play.py). There is no playback
and no robotic fallback: on failure the caller leaves the alert silent.

Microsoft is migrating Read Aloud to the api.msedgeservices.com host, which
rejects the legacy speech.platform.bing.com endpoint with 403 in several
regions while edge-tts still points at it. Synthesis therefore tries the new
host first and retries on the legacy one, so the natural voice keeps working.

The voice is passed by the caller (SOUND_VOICE, default en-US-AriaNeural).
"""

import argparse
import asyncio
import sys


def rate(value: str) -> str:
    try:
        number = int(float(value))
    except ValueError:
        number = 0
    number = max(-100, min(100, number))
    return f"{number:+d}%"


async def synth(text: str, path: str, voice: str, rate_value: str) -> None:
    import edge_tts
    import edge_tts.communicate as communicate
    from edge_tts.constants import TRUSTED_CLIENT_TOKEN

    # Edge 155 talks to api.msedgeservices.com; the legacy host increasingly
    # answers 403 (see rany2/edge-tts#490). Try the new host first, then the
    # library's own endpoint. Both accept the legacy Sec-MS-GEC token.
    new_host = (
        "wss://api.msedgeservices.com/tts/cognitiveservices/websocket/v1"
        f"?Ocp-Apim-Subscription-Key={TRUSTED_CLIENT_TOKEN}"
    )
    last_error = None
    for endpoint in (new_host, None):
        original = communicate.WSS_URL
        if endpoint:
            communicate.WSS_URL = endpoint
        try:
            communicator = edge_tts.Communicate(text, voice, rate=rate_value)
            await communicator.save(path)
            return
        except Exception as error:  # noqa: BLE001
            last_error = error
        finally:
            communicate.WSS_URL = original
    if last_error is None:
        raise RuntimeError("edge-tts failed without an error")
    raise last_error


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthesize stdin with Edge neural TTS.")
    parser.add_argument("--out", required=True, help="Write the MP3 to this path.")
    parser.add_argument("--voice", default="en-US-AriaNeural", help="Edge voice name.")
    parser.add_argument("--rate", default="0", help="-100..100, mapped to an Edge percentage.")
    args = parser.parse_args()

    text = sys.stdin.read().strip()
    if not text:
        return 0

    try:
        asyncio.run(synth(text, args.out, args.voice, rate(args.rate)))
    except Exception as error:  # noqa: BLE001
        print(f"edge-tts failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
