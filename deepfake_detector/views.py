from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
from io import BytesIO
from deepfake_detector.ai.inference import process_image
from asgiref.sync import async_to_sync, sync_to_async

# NPU 상태 관리
npu_state = {"current": "npu0"}  # 시작은 npu0

@sync_to_async
@csrf_exempt
@async_to_sync
async def inference(request):
    """
    Deepfake 판별 API

    Args:
        request: Django 요청 객체 (이미지 포함)

    Returns:
        JsonResponse: 추론 결과
    """
    if request.method == "POST" and request.FILES.get("image"):
        try:
            # 이미지 읽기
            image_file = request.FILES["image"]
            image = Image.open(BytesIO(image_file.read())).convert("RGB")

            # 추론 처리
            result = await process_image(image, npu_state)

            # 성공 응답 반환
            return JsonResponse(result)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)
