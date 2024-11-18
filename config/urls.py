from django.contrib import admin 
from django.urls import path
from deepfake_detector.views import inference  # API 뷰 임포트

urlpatterns = [
    # 다른 URL 패턴 예시
    path('admin/', admin.site.urls),

    # Inference API URL
    path('api/inference/', inference, name='inference'),
]
