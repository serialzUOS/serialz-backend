import logging
import tempfile
from asgiref.sync import async_to_sync
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, StreamingHttpResponse, HttpResponse
from PIL import Image
from io import BytesIO
from deepfake_detector.ai.inference import detect_and_crop_face, preprocess_image, process_with_npu, process_video, process_with_npu_fusionPE
import aiofiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

npu_state = {"current": "npu0"}  # NPU 상태 관리


def index(request):
    return HttpResponse("안녕하세요 환영합니다.")


@csrf_exempt
@async_to_sync
async def image_inference(request):
    """이미지 추론 API"""
    if request.method == "POST" and request.FILES.get("image"):
        try:
            image_file = request.FILES["image"]
            image = Image.open(BytesIO(image_file.read())).convert("RGB")

            # 얼굴 검출 및 크롭
            cropped_face = await detect_and_crop_face(image)
            if cropped_face is None:
                return JsonResponse({"error": "No face detected in the image."}, status=400)

            # 이미지 전처리
            input_tensor = await preprocess_image(cropped_face)

            # # NPU 상태 관리
            # current_npu = npu_state["current"]
            # npu_state["current"] = "npu1" if current_npu == "npu0" else "npu0"

            # # NPU 추론
            # result = await process_with_npu(input_tensor, current_npu)
            result = await process_with_npu_fusionPE(input_tensor)
            return JsonResponse(result)

        except Exception as e:
            logging.error(f"Error during image inference: {e}", exc_info=True)
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)


@csrf_exempt
@async_to_sync
async def video_inference(request):
    """비디오 추론 API"""
    if request.method == "POST" and request.FILES.get("video"):
        try:
            # 비디오 파일 저장
            video_file = request.FILES["video"]
            temp_video_path = tempfile.mktemp(suffix=".mp4")
            result_csv_path = tempfile.mktemp(suffix=".csv")

            with open(temp_video_path, "wb") as f:
                f.write(video_file.read())

            # 비디오 처리 및 추론
            await process_video(temp_video_path, result_csv_path, npu_state)

            # 결과 CSV를 스트리밍으로 반환
            async def file_iterator(file_path):
                async with aiofiles.open(file_path, mode="rb") as f:
                    while True:
                        chunk = await f.read(8192)
                        if not chunk:
                            break
                        yield chunk

            response = StreamingHttpResponse(
                file_iterator(result_csv_path),
                content_type="text/csv"
            )
            response["Content-Disposition"] = "attachment; filename=inference_results.csv"
            return response

        except Exception as e:
            logging.error(f"Error during video inference: {e}", exc_info=True)
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)
