from django.apps import AppConfig
import asyncio
from deepfake_detector.ai.startup import initialize_queues

class DeepfakeDetectorConfig(AppConfig):

    default_auto_field = "django.db.models.BigAutoField"
    name = "deepfake_detector"

    def ready(self):
        loop = asyncio.get_event_loop()
        loop.create_task(
            self.initialize_with_logging(
                "/home/ubuntu/serialz/serialz-backend/deepfake_detector/ai/optimized_model/ENTROPY_SYM_1.dfg"
            )
        )
        print("Queues initialization task created in background.")

    async def initialize_with_logging(self, model_path):
        print("Starting NPU queue initialization...")
        await initialize_queues(model_path)