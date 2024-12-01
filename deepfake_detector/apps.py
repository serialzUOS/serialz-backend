from django.apps import AppConfig
import asyncio
from deepfake_detector.ai.startup import initialize_queues
from deepfake_detector.ai.startup_fusion import initialize_fusion_queue

MODEL_PATH = "/home/ubuntu/serialz/serialz-backend/deepfake_detector/ai/optimized_model/SQNR_ASYM.dfg"

class DeepfakeDetectorConfig(AppConfig):

    default_auto_field = "django.db.models.BigAutoField"
    name = "deepfake_detector"

    def ready(self):
        loop = asyncio.get_event_loop()
        loop.create_task(
            self.initialize_with_logging(MODEL_PATH)
        )
        print("Queues initialization task created in background.")

    async def initialize_with_logging(self, model_path):
        print("Starting NPU queue initialization...")
        # await initialize_queues(model_path)
        await initialize_fusion_queue(model_path)