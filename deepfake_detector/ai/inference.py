import numpy as np
import torchvision.transforms as transforms
from deepfake_detector.ai.startup import queues
from facenet_pytorch.models.mtcnn import MTCNN
import torch
import cv2
import csv
import logging
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# MTCNN 초기화
face_detector = MTCNN(
    margin=0,
    thresholds=[0.60, 0.60, 0.60],
    device="cuda" if torch.cuda.is_available() else "cpu",
    select_largest=True,
    keep_all=True,)


# 이미지 전처리 함수
async def preprocess_image(image):
    """
    이미지를 Furiosa 입력 형식으로 전처리합니다.

    Args:
        image (PIL.Image): 입력 이미지

    Returns:
        numpy.ndarray: 전처리된 이미지
    """
    IMG_SIZE = 299
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    tensor = transform(image).unsqueeze(0)  # 배치 차원 추가
    return tensor.numpy().astype(np.float32)


# Softmax 계산 함수
def softmax(logits):
    max_logits = np.max(logits)  # 로그 항등성을 위해 최댓값 계산
    exp_logits = np.exp(logits - max_logits)
    return exp_logits / np.sum(exp_logits)


async def detect_and_crop_face(image):
    """
    주어진 이미지에서 얼굴을 감지하고 크롭합니다.

    Args:
        image (PIL.Image): 입력 이미지

    Returns:
        PIL.Image or None: 감지된 얼굴 영역을 크롭한 이미지 또는 None
    """
    frame_rgb = np.array(image)
    frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    boxes, _ = face_detector.detect(image, landmarks=False)

    if boxes is None:
        logging.warning("No face detected.")
        return None

    xmin, ymin, xmax, ymax = map(int, boxes[0])
    face_crop = frame_rgb[ymin:ymax, xmin:xmax]
    return Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))


async def process_with_npu(input_tensor, current_npu):
    """
    NPU에서 데이터를 처리하고 결과를 반환합니다.

    Args:
        input_tensor (torch.Tensor): 전처리된 입력 데이터
        current_npu (str): 사용할 NPU ("npu0" 또는 "npu1")

    Returns:
        dict: 추론 결과
    """
    submitter, receiver = queues[current_npu]

    # 데이터 제출 및 결과 수신
    await submitter.submit(input_tensor)
    async for _, outputs in receiver:
        probabilities = softmax(outputs[1][0])
        return {
            "npu_used": current_npu,
            "deepfake_probability": float(probabilities[1]),
            "normal_probability": float(probabilities[0]),
        }


async def process_video(video_path, output_path, npu_state):
    """
    비디오를 처리하고 추론 결과를 CSV로 저장합니다.

    Args:
        video_path (str): 비디오 파일 경로
        output_path (str): 결과 CSV 파일 경로
        npu_state (dict): 현재 NPU 상태를 나타내는 사전 {"current": "npu0" or "npu1"}

    Returns:
        str: 생성된 CSV 파일 경로
    """
    capture = cv2.VideoCapture(video_path)
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(f"Video FPS: {fps}, Frame Count: {frame_count}")

    space = max(1, fps // 3)
    logging.info(f"Frame interval: {space}")

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "deepfake_probability"])

        for idx in range(0, frame_count, space):
            logging.info(f"Processing frame {idx}/{frame_count}")
            capture.set(cv2.CAP_PROP_POS_FRAMES, idx)

            # 프레임 읽기
            success, frame = capture.read()
            if not success or frame is None:
                logging.warning(
                    f"Frame {idx} could not be read or is None.")
                continue

            # 얼굴 감지 및 크롭
            frame_image = Image.fromarray(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cropped_face = await detect_and_crop_face(frame_image)
            if cropped_face is None:
                logging.warning(f"No face detected in frame {idx}.")
                continue

            # 이미지 전처리
            input_tensor = await preprocess_image(cropped_face)

            # NPU 상태 관리 및 추론
            current_npu = npu_state["current"]
            npu_state["current"] = "npu1" if current_npu == "npu0" else "npu0"
            result = await process_with_npu(input_tensor, current_npu)

            writer.writerow([idx / fps, result["deepfake_probability"]])
            logging.info(f"Frame {idx} processed successfully.")

    logging.info("Video processing completed.")
