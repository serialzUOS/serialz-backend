import os
import django
from django.core.asgi import get_asgi_application
from deepfake_detector.ai.startup import initialize_queues
from channels.routing import ProtocolTypeRouter

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

# ASGI application
django_asgi_app = get_asgi_application()

# ASGI Middleware로 초기화

class AsyncStartupMiddleware:
    def __init__(self, app):
        self.app = app
        self.initialized = False

    async def __call__(self, scope, receive, send):
        if not self.initialized:
            await initialize_queues("/home/ubuntu/serialz/serialz-backend/deepfake_detector/ai/optimized_model/ENTROPY_SYM_1.dfg")
            self.initialized = True
        return await self.app(scope, receive, send)

application = ProtocolTypeRouter(
    {
        "http": AsyncStartupMiddleware(django_asgi_app),
    }
)
