import logging
import io
from shared.config import config
from faster_whisper import WhisperModel
from typing import Union, BinaryIO, cast

logger = logging.getLogger(__name__)


class STTProvider:
    """
    Provides local Speech-to-Text transcription using faster-whisper.
    """

    def __init__(self, config_instance=config):
        self.config = config_instance
        self.model_size = self.config.STT_MODEL_SIZE
        self.device = self.config.STT_DEVICE
        self.compute_type = self.config.STT_COMPUTE_TYPE
        self.model = self._init_model()

    def _init_model(self) -> WhisperModel:
        """
        Initializes and returns the Whisper model.
        """
        logger.info(
            f"Loading local Whisper model: {self.model_size} on {self.device}..."
        )
        try:
            model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
            logger.info("Local Whisper model initialized successfully.")
            return model
        except Exception as e:
            logger.critical(f"Failed to initialize Whisper model: {e}")
            raise

    def transcribe(self, audio_input: Union[str, BinaryIO, bytes, bytearray, memoryview]) -> str:
        """
        Transcribes audio from a file path, a file-like object (BytesIO), or raw bytes.
        """
        try:
            # faster-whisper's transcribe() supports file-like objects.
            # We explicitly check/convert if needed, though the library is robust.
            if isinstance(audio_input, (bytes, bytearray, memoryview)):
                audio_input = io.BytesIO(bytes(audio_input))
            # If it's already a file path (str) or a BinaryIO, leave it as-is.

            # beam_size=5 is a standard tradeoff for accuracy vs speed
            # Cast to the union accepted by faster-whisper to satisfy static type checkers
            audio_arg = cast(Union[str, BinaryIO], audio_input)
            segments, info = self.model.transcribe(audio_arg, beam_size=5)

            logger.info(
                f"Detected language '{info.language}' ({info.language_probability:.2f})"
            )

            # Collect text segments
            text_segments = [segment.text for segment in segments]
            transcribed_text = " ".join(text_segments).strip()

            return transcribed_text

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise