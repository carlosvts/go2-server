"""Leitura do WAV recebido da TV Box."""

import io
import wave

import numpy as np

SAMPLE_RATE = 16000


class InvalidAudioError(ValueError):
    """O arquivo não é um WAV mono 16 kHz PCM16 com áudio."""


def decode_wav(data: bytes) -> np.ndarray:
    """PCM16 mono 16 kHz → float32 em [-1, 1]."""
    if not data:
        raise InvalidAudioError("áudio vazio")
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            layout = (wav.getnchannels(), wav.getsampwidth(), wav.getframerate())
            frames = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as error:
        raise InvalidAudioError(f"não é um WAV PCM válido ({error})") from error
    if layout != (1, 2, SAMPLE_RATE):
        raise InvalidAudioError(
            f"esperado WAV mono 16 kHz PCM16; veio canais={layout[0]} "
            f"bytes_por_amostra={layout[1]} taxa={layout[2]}"
        )
    if not frames:
        raise InvalidAudioError("WAV sem amostras")
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
