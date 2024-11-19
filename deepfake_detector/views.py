from django.views.decorators.csrf import csrf_exempt
from PIL import Image
from io import BytesIO
from django.http import JsonResponse, StreamingHttpResponse
import tempfile
from asgiref.sync import async_to_sync, sync_to_async
from deepfake_detector.ai.inference import preprocess_image, detect_and_crop_face, process_with_npu, \
    process_video
import aiofiles

npu_state = {"current": "npu0"}

@csrf_exempt
@async_to_sync
async def image_inference(request):
    if request.method == "POST" and request.FILES.get("image"):
        try:
            # 이미지 읽기
            image_file = request.FILES["image"]
            image = Image.open(BytesIO(image_file.read())).convert("RGB")
            
            # 얼굴 감지 및 크롭
            cropped_face = await detect_and_crop_face(image)
            if cropped_face is None:
                return JsonResponse({"error": "No face detected in the image."}, status=400)

            # 이미지 전처리
            input_tensor = await preprocess_image(cropped_face)

            # NPU 상태 관리
            current_npu = npu_state["current"]
            npu_state["current"] = "npu1" if current_npu == "npu0" else "npu0"

            # NPU 추론
            result = await process_with_npu(input_tensor, current_npu)

            return JsonResponse(result)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)


@csrf_exempt
@async_to_sync
async def video_inference(request):
    if request.method == "POST" and request.FILES.get("video"):
        try:
            # 비디오 읽기 및 임시 파일 경로 생성
            video_file = request.FILES["video"]
            temp_video_path = tempfile.mktemp(suffix=".mp4")
            result_csv_path = tempfile.mktemp(suffix=".csv")

            with open(temp_video_path, "wb") as f:
                f.write(video_file.read())

            # 비디오 처리 및 추론
            await process_video(temp_video_path, result_csv_path, npu_state)

            # 비동기 파일 스트리밍
            async def file_iterator(file_path):
                async with aiofiles.open(file_path, mode="rb") as f:
                    while chunk := await f.read(8192):
                        yield chunk

            # 비동기 스트리밍 HTTP 응답
            response = StreamingHttpResponse(
                file_iterator(result_csv_path),
                content_type="text/csv"
            )
            response["Content-Disposition"] = "attachment; filename=inference_results.csv"
            return response

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)