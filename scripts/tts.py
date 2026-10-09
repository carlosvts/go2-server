"""Fala sintética para os scripts de teste: WAV mono 16 kHz PCM16.

Usa as vozes do Piper que estiverem em `models/piper/*.onnx` (bem mais
naturais) e, sem elas, o espeak-ng. Para baixar duas vozes pt-BR:

    base=https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR
    for v in faber cadu; do for ext in onnx onnx.json; do
      curl -L --create-dirs -o models/piper/pt_BR-$v-medium.$ext \
        $base/$v/medium/pt_BR-$v-medium.$ext
    done; done
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ESPEAK_VOICES = (("pt-br", 150), ("pt-br+f3", 140), ("pt-br+m3", 165), ("pt-br+f2", 155))


def piper_voices() -> list[Path]:
    return sorted((ROOT / "models" / "piper").glob("*.onnx"))


def engine_name() -> str:
    return "piper" if piper_voices() else "espeak-ng"


def synthesize(text: str, index: int = 0) -> bytes:
    """`index` alterna a voz, para não medir uma só."""
    voices = piper_voices()
    with tempfile.TemporaryDirectory() as folder:
        raw, out = Path(folder) / "raw.wav", Path(folder) / "out.wav"
        if voices:
            piper = ["piper"] if shutil.which("piper") else ["uvx", "--from", "piper-tts", "piper"]
            subprocess.run(
                [*piper, "-m", str(voices[index % len(voices)]), "-f", str(raw)],
                input=text.encode(),
                check=True,
                capture_output=True,
            )
        else:
            voice, speed = ESPEAK_VOICES[index % len(ESPEAK_VOICES)]
            subprocess.run(
                ["espeak-ng", "-v", voice, "-s", str(speed), "-w", str(raw), text], check=True
            )
        subprocess.run(
            ["sox", str(raw), "-r", "16000", "-c", "1", "-b", "16", str(out), "pad", "0.3", "0.3"],
            check=True,
        )
        return out.read_bytes()
