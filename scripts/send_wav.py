"""Envia uma frase ao servidor, para testar sem a TV Box.

    uv run python scripts/send_wav.py frase.wav
    uv run python scripts/send_wav.py --say "hey jarvis senta" --reason wake_word
    uv run python scripts/send_wav.py --cancel stop

`--say` sintetiza a frase (Piper se houver voz em models/piper, senão espeak-ng; e sox). Com
`--reason wake_word` os campos `local_*` vão nulos, como no modo thin.
"""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tts import synthesize  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("wav", nargs="?", type=Path, help="WAV mono 16 kHz PCM16")
    parser.add_argument("--say", help="sintetiza esta frase em vez de ler um arquivo")
    parser.add_argument("--cancel", choices=["stop", "local_command"], help="chama /v1/cancel")
    parser.add_argument("--server", default="http://127.0.0.1:9000")
    parser.add_argument("--edge-id", default="send-wav")
    parser.add_argument("--reason", default="unk", help="unk, low_conf, too_long ou wake_word")
    parser.add_argument("--hypothesis", help="local_hypothesis (modo edge)")
    parser.add_argument("--confidence", type=float, help="local_confidence (modo edge)")
    parser.add_argument("--utterance-id", help="padrão: um UUID novo")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    with httpx.Client(base_url=args.server, timeout=args.timeout) as client:
        if args.cancel:
            response = client.post(
                "/v1/cancel", json={"edge_id": args.edge_id, "reason": args.cancel}
            )
            print(response.status_code, response.text)
            return 0 if response.is_success else 1

        if args.say:
            audio = synthesize(args.say)
        elif args.wav:
            audio = args.wav.read_bytes()
        else:
            parser.error("informe um WAV, --say ou --cancel")
        thin = args.reason == "wake_word"
        meta = {
            "utterance_id": args.utterance_id or str(uuid.uuid4()),
            "edge_id": args.edge_id,
            "reason": args.reason,
            "local_hypothesis": None if thin else args.hypothesis,
            "local_confidence": None if thin else args.confidence,
        }
        started = time.perf_counter()
        response = client.post(
            "/v1/utterance",
            files={"audio": ("utterance.wav", audio, "audio/wav")},
            data={"meta": json.dumps(meta, ensure_ascii=False)},
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
    print(f"HTTP {response.status_code} em {elapsed_ms:.0f} ms")
    try:
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    except ValueError:
        print(response.text)
    return 0 if response.is_success else 1


if __name__ == "__main__":
    sys.exit(main())
