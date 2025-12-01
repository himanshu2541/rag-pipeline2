import grpc
from concurrent import futures
import io
import logging

from shared.config import config, setup_logging

from stt_service.providers.stt import STTProvider

from protos import service_pb2, service_pb2_grpc

setup_logging()
logger = logging.getLogger("STTService")


class TranscriberServicer(service_pb2_grpc.TranscriberServicer):
    """
    gRPC Servicer for Speech-to-Text Transcription.
    """

    def __init__(self, config_instance=config):
        self.config = config_instance
        self.provider = STTProvider(self.config)

    def Transcribe(self, request, context):
        """
        Handles the Transcribe gRPC call.
        Takes raw bytes from request.audio_content and processes them in-memory.
        """

        # Dynamically resolve the response message class from the generated protos
        def _get_response_cls():
            # common candidate names to try
            candidates = (
                "TranscriptionResponse",
                "TranscribeResponse",
                "TranscriptionReply",
                "TranscriptionResult",
                "Transcript",
            )
            for name in candidates:
                cls = getattr(service_pb2, name, None)
                if cls is not None:
                    return cls
            # fallback: try to find any attribute with 'transcrip'/'transcript' in the name
            for attr in dir(service_pb2):
                low = attr.lower()
                if "transcrip" in low or "transcript" in low:
                    cls = getattr(service_pb2, attr)
                    try:
                        # verify it can be instantiated as a message
                        _ = cls()
                        return cls
                    except Exception:
                        continue
            raise AttributeError(
                "No response message class found in protos.service_pb2; "
                "expected one of TranscriptionResponse, TranscribeResponse, TranscriptionReply, TranscriptionResult, Transcript"
            )

        response_cls = _get_response_cls()

        try:
            logger.info(f"Received audio chunk: {len(request.audio_content)} bytes")

            # 1. Convert raw bytes to a file-like object for faster-whisper
            audio_stream = io.BytesIO(request.audio_content)

            # 2. Run Transcription (No disk I/O)
            text = self.provider.transcribe(audio_stream)

            logger.info(f"Transcribed: {text[:30]}...")

            # 3. Return Response using the discovered response class
            return response_cls(text=text, success=True, error_message="")

        except Exception as e:
            logger.error(f"Error during transcription: {e}", exc_info=True)
            return response_cls(text="", success=False, error_message=str(e))


def serve():
    # Typically 1 worker is sufficient if running on a single GPU
    # as the model lock prevents parallel inference anyway.
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    service_pb2_grpc.add_TranscriberServicer_to_server(TranscriberServicer(), server)
    server_address = f"{config.STT_SERVICE_HOST}:{config.STT_SERVICE_PORT}"
    server.add_insecure_port(server_address)
    logger.info(f"Starting STT gRPC server on {server_address}...")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
