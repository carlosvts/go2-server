"""faster-whisper em modo livre, pt-BR, com as defesas contra alucinação.

O Whisper inventa texto em silêncio e ruído ("Obrigado por assistir",
"Legendas pela comunidade Amara.org"). As defesas, na ordem:

1. VAD (Silero, embutido): áudio sem voz nem chega ao modelo;
2. `condition_on_previous_text=False` e temperatura 0: sem contexto para
   arrastar e sem amostragem;
3. `no_speech_prob` alta com `avg_logprob` baixa ⇒ rejeita;
4. `avg_logprob` baixa ou texto repetitivo (taxa de compressão) ⇒ rejeita;
5. transcrição vazia ou na lista de alucinações conhecidas ⇒ rejeita.
"""

import numpy as np

from app.interpret.text import fold, tokenize, words
from app.transcribe.base import Transcription


def text_key(text: str) -> str:
    return " ".join(fold(word) for word in words(tokenize(text)))


class WhisperTranscriber:
    def __init__(
        self,
        model: str = "small",
        compute_type: str = "int8",
        cpu_threads: int = 4,
        num_workers: int = 2,
        beam_size: int = 5,
        no_speech_threshold: float = 0.6,
        min_avg_logprob: float = -1.0,
        max_compression_ratio: float = 2.4,
        hallucinations: list[str] | None = None,
    ) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            model,
            device="cpu",
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
        )
        self._beam_size = beam_size
        self._no_speech_threshold = no_speech_threshold
        self._min_avg_logprob = min_avg_logprob
        self._max_compression_ratio = max_compression_ratio
        self._hallucinations = {text_key(text) for text in hallucinations or []}

    def transcribe(self, audio: np.ndarray) -> Transcription:
        segments, _ = self._model.transcribe(
            audio,
            language="pt",
            task="transcribe",
            beam_size=self._beam_size,
            temperature=0.0,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            # As rejeições são feitas abaixo, com o motivo no log, em vez de o
            # modelo descartar segmentos em silêncio.
            no_speech_threshold=None,
            log_prob_threshold=None,
            compression_ratio_threshold=None,
        )
        segments = list(segments)
        if not segments:
            return Transcription("", "no_speech", {"vad": "sem voz"})

        text = " ".join(segment.text.strip() for segment in segments).strip()
        tokens = [max(len(segment.tokens), 1) for segment in segments]
        avg_logprob = float(np.average([s.avg_logprob for s in segments], weights=tokens))
        no_speech_prob = max(float(segment.no_speech_prob) for segment in segments)
        compression_ratio = max(float(segment.compression_ratio) for segment in segments)
        details = {
            "avg_logprob": round(avg_logprob, 3),
            "no_speech_prob": round(no_speech_prob, 3),
            "compression_ratio": round(compression_ratio, 2),
        }

        reason = None
        if not text_key(text):
            reason = "empty_transcript"
        elif no_speech_prob > self._no_speech_threshold and avg_logprob < self._min_avg_logprob:
            reason = "no_speech"
        elif avg_logprob < self._min_avg_logprob:
            reason = "low_confidence"
        elif (
            compression_ratio > self._max_compression_ratio
            or text_key(text) in self._hallucinations
        ):
            reason = "hallucination"
        return Transcription(text, reason, details)
