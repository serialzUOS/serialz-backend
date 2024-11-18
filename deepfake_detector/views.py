from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
from io import BytesIO
from deepfake_detector.ai.inference import preprocess_image, softmax
from asgiref.sync import async_to_sync, sync_to_async
from deepfake_detector.ai.startup import queues

# NPU 상태 관리
npu_state = {"current": "npu0"}  # 시작은 npu0

@sync_to_async
@csrf_exempt
@async_to_sync
async def inference(request):
    """
    Deepfake 판별 API
    Args:
        request: Django 요청 객체
    Returns:
        JsonResponse: 추론 결과
    """
    if request.method == "POST" and request.FILES.get("image"):
        try:
            # 이미지 읽기 및 전처리
            image_file = request.FILES["image"]
            image = Image.open(BytesIO(image_file.read())).convert("RGB")
            input_tensor = await preprocess_image(image)

            # 현재 사용 중인 NPU 선택
            current_npu = npu_state["current"]
            npu_state["current"] = "npu1" if current_npu == "npu0" else "npu0"

            # 전역 큐에서 NPU 사용
            queue = queues[current_npu]
            submitter, receiver = queue

            # 입력 데이터 제출 및 결과 수신
            await submitter.submit(input_tensor)
            async for _, outputs in receiver:
                probabilities = softmax(outputs[1][0])
                return JsonResponse({
                    "npu_used": current_npu,
                    "deepfake_probability": float(probabilities[1]),
                    "normal_probability": float(probabilities[0]),
                })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)
