import numpy as np
import torchvision.transforms as transforms
from deepfake_detector.ai.startup import queues
from deepfake_detector.ai.startup_fusion import fusion_queue
from facenet_pytorch.models.mtcnn import MTCNN
import torch
import cv2
import logging
from PIL import Image
import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiofiles
import aiocsv
from deepfake_detector.ai.startup import is_initialized
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# MTCNN 초기화
face_detector = MTCNN(
    margin=0,
    thresholds=[0.60, 0.60, 0.60],
    device="cuda" if torch.cuda.is_available() else "cpu",
    select_largest=True,
    keep_all=True,
)

executor = ThreadPoolExecutor()

# 비동기 VideoCapture read
async def async_capture_read(capture):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, capture.read)

# 비동기 MTCNN detect
async def async_detect(image):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: face_detector.detect(image, landmarks=False))

async def preprocess_image(image):
    IMG_SIZE = 288
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    tensor = transform(image).unsqueeze(0)
    return tensor.numpy().astype(np.float32)

def softmax(x,axis=0):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

async def detect_and_crop_face(image):
    frame_rgb = np.array(image)
    frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    boxes, _ = await async_detect(image)

    if boxes is None:
        logging.warning("No face detected.")
        return None

    xmin, ymin, xmax, ymax = map(int, boxes[0])
    face_crop = frame_rgb[ymin:ymax, xmin:xmax]
    return Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))

async def process_with_npu(input_tensor, current_npu):

    # 디버깅 정보 출력
    logging.info(f"Queues: {queues}")
    logging.info(f"Current NPU: {current_npu}")
    
    # current_npu가 queues에 없는 경우 처리
    if current_npu not in queues:
        logging.error(f"Current NPU '{current_npu}' not found in queues.")
        raise ValueError(f"Current NPU '{current_npu}' not found in queues.")
    
    # queues[current_npu]가 None인 경우 처리
    if queues[current_npu] is None:
        logging.error(f"Queues[{current_npu}] is None.")
        raise ValueError(f"Queues[{current_npu}] is None.")

    # 정상적으로 queues에서 값을 가져오는 경우
    submitter, receiver = queues[current_npu]

    await submitter.submit(input_tensor)
    async for _, outputs in receiver:
        logging.info(f"Deepfake probability: {outputs[1][0]}")
        probabilities = softmax(outputs[1][0])
        deepfake_probability = round(probabilities[1] * 100, 2)
        normal_probability = round(probabilities[0] * 100, 2)
        logging.info(f"Deepfake probability: {deepfake_probability}")
        return {
            "npu_used": current_npu,
            "deepfake_probability": deepfake_probability,
            "normal_probability": normal_probability,
        }


async def process_video(video_path, output_path, npu_state):
    
    capture = cv2.VideoCapture(video_path)
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    space = max(1, fps)  # 1초에 3프레임 간격으로 처리

    async with aiofiles.open(output_path, mode="w", newline="") as csvfile:
        writer = aiocsv.AsyncWriter(csvfile)  # await 제거
        await writer.writerow(["timestamp", "deepfake_probability"])  # 헤더 작성

        for idx in range(0, frame_count, space):
            capture.set(cv2.CAP_PROP_POS_FRAMES, idx)

            # 비동기 프레임 읽기
            success, frame = await async_capture_read(capture)
            if not success or frame is None:
                logging.warning(f"Frame {idx} could not be read or is None.")
                continue

            # 얼굴 감지 및 크롭
            frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cropped_face = await detect_and_crop_face(frame_image)
            if cropped_face is None:
                logging.warning(f"No face detected in frame {idx}.")
                continue

            # 이미지 전처리
            input_tensor = await preprocess_image(cropped_face)

            # NPU 상태 관리 및 추론
            # current_npu = npu_state["current"]
            # npu_state["current"] = "npu1" if current_npu == "npu0" else "npu0"
            # result = await process_with_npu(input_tensor, current_npu)

            result = await process_with_npu_fusionPE(input_tensor)

            # 결과를 CSV에 작성
            await writer.writerow([idx / fps, result["deepfake_probability"]])
            logging.info(f"Frame {idx} processed successfully.")

async def process_with_npu_fusionPE(input_tensor):

    logging.info(f"Using FusionPE queue for inference.")
    
    # FusionPE 큐를 사용하는 방식으로 변경
    if fusion_queue is None:
        logging.error("FusionPE queue is not initialized.")
        raise ValueError("FusionPE queue is not initialized.")
    
    submitter, receiver = fusion_queue  # FusionPE 큐에서 submitter와 receiver 가져오기

    await submitter.submit(input_tensor)
    async for _, outputs in receiver:
        logging.info(f"Deepfake probability: {outputs[1][0]}")
        probabilities = softmax(outputs[1][0])
        deepfake_probability = round(probabilities[1] * 100, 2)
        normal_probability = round(probabilities[0] * 100, 2)
        logging.info(f"Deepfake probability: {deepfake_probability}")
        return {
            "deepfake_probability": deepfake_probability,
            "normal_probability": normal_probability,
        }
