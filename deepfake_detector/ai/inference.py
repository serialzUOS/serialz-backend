import logging
import cv2
from PIL import Image
import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiofiles
import aiocsv
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import numpy as np
import torch
from facenet_pytorch.models.mtcnn import MTCNN


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

executor = ThreadPoolExecutor()

# MTCNN 초기화 (전역 변수로 설정)
face_detector = MTCNN(
    margin=0,
    thresholds=[0.60, 0.60, 0.60],
    device="cuda" if torch.cuda.is_available() else "cpu",
    select_largest=True,
    keep_all=False,  # 한 얼굴만 감지
)

# 얼굴 검출 및 크롭
async def detect_and_crop_face(image):
    frame_rgb = np.array(image)
    frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    boxes, _ = await asyncio.get_event_loop().run_in_executor(
        executor, lambda: face_detector.detect(frame_rgb)
    )
    if boxes is None:
        logging.warning("No face detected.")
        return None

    xmin, ymin, xmax, ymax = map(int, boxes[0])
    face_crop = frame_rgb[ymin:ymax, xmin:xmax]
    return Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))


# 이미지 전처리
async def preprocess_image(image):
    IMG_SIZE = 299
    transform = Compose([
        Resize((IMG_SIZE, IMG_SIZE)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(image).unsqueeze(0)
    return tensor.numpy().astype(np.float32)


# NPU 추론
async def process_with_npu(input_tensor, current_npu):
    probabilities = np.random.rand(1, 2)  # 예제용 더미 데이터
    probabilities = softmax(probabilities[0])
    deepfake_probability = round(probabilities[1] * 100, 2)
    return {
        "npu_used": current_npu,
        "deepfake_probability": deepfake_probability,
    }


# Softmax 계산
def softmax(logits):
    logits = torch.tensor(logits)
    max_logits = logits.max()
    exp_logits = torch.exp(logits - max_logits)
    return (exp_logits / exp_logits.sum()).cpu().numpy()


# 비디오 처리 및 병렬 추론
async def process_video_and_generate_csv(video_path, output_csv_path, npu_state):
    capture = cv2.VideoCapture(video_path)
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # 30fps로 처리
    space = max(1, fps // 30)
    frames = []
    for idx in range(0, frame_count, space):
        capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = capture.read()
        if not ret or frame is None:
            continue
        frames.append((idx / fps, frame))
    capture.release()

    logging.info(f"Extracted {len(frames)} frames for processing.")

    # 얼굴 검출 및 전처리
    async def process_frame(frame_tuple):
        timestamp, frame = frame_tuple
        frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cropped_face = await detect_and_crop_face(frame_image)
        if cropped_face is None:
            return None
        input_tensor = await preprocess_image(cropped_face)

        current_npu = npu_state["current"]
        npu_state["current"] = "npu1" if current_npu == "npu0" else "npu0"

        result = await process_with_npu(input_tensor, current_npu)
        return (timestamp, result["deepfake_probability"])

    tasks = [process_frame(frame) for frame in frames]
    results = await asyncio.gather(*tasks)

    # 결과 CSV로 저장
    async with aiofiles.open(output_csv_path, mode="w", newline="") as csvfile:
        writer = aiocsv.AsyncWriter(csvfile)
        await writer.writerow(["timestamp", "deepfake_probability"])
        for result in results:
            if result is not None:
                await writer.writerow(result)
