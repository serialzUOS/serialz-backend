from django.views.decorators.csrf import csrf_exempt
from PIL import Image
from io import BytesIO
import numpy as np
import os
import csv
from django.http import JsonResponse, FileResponse
from facenet_pytorch.models.mtcnn import MTCNN
import tempfile
import cv2
from asgiref.sync import async_to_sync, sync_to_async
import torch
import asyncio
from deepfake_detector.ai.inference import preprocess_image, softmax
from deepfake_detector.ai.startup import queues
from tqdm.asyncio import tqdm
import logging
# NPU 상태 관리
npu_state = {"current": "npu0"}
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# MTCNN 초기화
face_detector = MTCNN(
    margin=0,
    thresholds=[0.60, 0.60, 0.60],
    device="cuda" if torch.cuda.is_available() else "cpu",
    select_largest=True,
    keep_all=True,)


@sync_to_async
@csrf_exempt
@async_to_sync
async def image_inference(request):
    """
    단일 이미지 추론 API
    """
    if request.method == "POST" and request.FILES.get("image"):
        try:
            # 이미지 읽기
            image_file = request.FILES["image"]
            image = Image.open(BytesIO(image_file.read())).convert("RGB")
            # 이미지 전처리
            input_tensor = await preprocess_image(image)
            # NPU 교대 처리
            current_npu = npu_state["current"]
            npu_state["current"] = "npu1" if current_npu == "npu0" else "npu0"
            
            # 전역 큐에서 NPU 사용
            submitter, receiver = queues[current_npu]

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

@sync_to_async
@csrf_exempt
@async_to_sync
async def video_inference(request):
    """
    비디오 파일 추론 API
    """
    if request.method == "POST" and request.FILES.get("video"):
        try:
            # 비디오 읽기
            video_file = request.FILES["video"]
            temp_video_path = tempfile.mktemp(suffix=".mp4")
            with open(temp_video_path, "wb") as f:
                f.write(video_file.read())

            # 비디오 처리 및 추론
            result_csv = await process_video(temp_video_path)

            # CSV 파일 응답
            response = FileResponse(
                open(result_csv, "rb"),
                as_attachment=True,
                filename="inference_results.csv",
            )
            return response

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)


async def process_video(video_path):
    """
    비디오를 처리하고 추론 결과 CSV 생성
    """
    capture = cv2.VideoCapture(video_path)
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(f"Video FPS: {fps}, Frame Count: {frame_count}")

    space = max(1, fps // 3)
    logging.info(f"Frame interval: {space}")

    csv_path = tempfile.mktemp(suffix=".csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "deepfake_probability"])

        for idx in range(0, frame_count, space):
            logging.info(f"Processing frame {idx}/{frame_count}")
            capture.set(cv2.CAP_PROP_POS_FRAMES, idx)

            # 프레임 읽기
            success, frame = capture.read()
            if not success or frame is None:
                logging.warning(f"Frame {idx} could not be read or is None.")
                continue

            # 얼굴 감지
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_image = Image.fromarray(frame_rgb)
            boxes, probs = face_detector.detect(frame_image, landmarks=False)
            if boxes is None:
                logging.warning(f"No face detected in frame {idx}.")
                continue

            # 크롭 및 추론
            xmin, ymin, xmax, ymax = map(int, boxes[0])
            face_crop = frame_rgb[ymin:ymax, xmin:xmax]
            input_tensor = await preprocess_image(Image.fromarray(face_crop))

            # NPU 처리
            current_npu = npu_state["current"]
            npu_state["current"] = "npu1" if current_npu == "npu0" else "npu0"
            submitter, receiver = queues[current_npu]

            # NPU에 데이터 제출 및 결과 수신
            await submitter.submit(input_tensor)
            async for task_id, outputs in receiver:
                probabilities = softmax(outputs[1][0])
                writer.writerow([idx / fps, float(probabilities[1])])
                logging.info(f"Frame {idx} processed successfully.")
                break  # 수신이 완료되면 루프를 종료합니다.

    logging.info("Video processing completed.")
    return csv_path