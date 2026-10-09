"""Vosk em modo livre (sem gramática), para comparação e como alternativa."""

import json
from pathlib import Path

import numpy as np

from app.audio import SAMPLE_RATE
from app.transcribe.base import Transcription


class VoskTranscriber:
    def __init__(self, model_path: Path) -> None:
        from vosk import Model, SetLogLevel

        if not model_path.is_dir():
            raise FileNotFoundError(f"Modelo Vosk não encontrado em {model_path}.")
        SetLogLevel(-1)
        self._model = Model(str(model_path))

    def transcribe(self, audio: np.ndarray) -> Transcription:
        from vosk import KaldiRecognizer

        # Um reconhecedor por chamada: ele guarda estado e não é thread-safe.
        recognizer = KaldiRecognizer(self._model, SAMPLE_RATE)
        recognizer.SetWords(True)
        recognizer.AcceptWaveform((audio * 32767).astype(np.int16).tobytes())
        result = json.loads(recognizer.FinalResult())
        text = result.get("text", "").strip()
        confidences = [word["conf"] for word in result.get("result", [])]
        details = {"min_conf": round(min(confidences), 3)} if confidences else {}
        return Transcription(text, None if text else "no_speech", details)
