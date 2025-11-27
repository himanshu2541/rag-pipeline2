import logging
import os
from faster_whisper import WhisperModel
from config import config

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
        
        logger.info(f"Loading local Whisper model: {self.model_size} on {self.device}...")
        try:
            self.model = WhisperModel(
                self.model_size, 
                device=self.device, 
                compute_type=self.compute_type
            )
            logger.info("Local Whisper model loaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to load Whisper model: {e}")
            raise

    def transcribe(self, file_path: str) -> str:
        """
        Transcribes an audio file to text.
        """
        try:
            segments, info = self.model.transcribe(file_path, beam_size=5)
            
            logger.info(f"Detected language '{info.language}' with probability {info.language_probability}")

            text_segments = [segment.text for segment in segments]
            transcribed_text = " ".join(text_segments).strip()
            
            return transcribed_text
            
        except Exception as e:
            logger.error(f"Transcription failed for {file_path}: {e}")
            raise