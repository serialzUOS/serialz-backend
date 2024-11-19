import numpy as np
import torchvision.transforms as transforms
from furiosa.runtime import session, create_queue
from asyncio import Lock

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
    exp_logits = np.exp(logits) 
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

