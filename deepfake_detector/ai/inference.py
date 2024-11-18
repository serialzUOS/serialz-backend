import numpy as np
import torchvision.transforms as transforms
from furiosa.runtime import session, create_queue
from asyncio import Lock

# 모델 경로 설정
MODEL_PATH = "/home/ubuntu/serialz/serialz-backend/deepfake_detector/ai/optimized_model/ENTROPY_SYM_1.dfg"

# NPU 상태 관리
npu_locks = {"npu0": Lock(), "npu1": Lock()}

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

# 추론 처리 함수
async def process_image(image, npu_state):
    """
    이미지에 대한 추론을 수행합니다.

    Args:
        image (PIL.Image): 입력 이미지
        npu_state (dict): NPU 상태 관리 변수

    Returns:
        dict: 추론 결과
    """
    # 현재 사용 중인 NPU와 Lock 가져오기
    current_npu = npu_state["current"]
    lock = npu_locks[current_npu]

    # NPU 동기화 및 상태 전환
    async with lock:
        npu_state["current"] = "npu1" if current_npu == "npu0" else "npu0"

        # 이미지 전처리
        input_tensor = await preprocess_image(image)
        print("이미지 전처리 완료")
        # NPU에서 추론 실행
        result = await run_inference(input_tensor, current_npu)

        # 결과 반환
        return {
            "npu_used": current_npu,
            **result,  # 추론 결과 포함
        }

async def run_inference(input_tensor, npu):
    """
    특정 NPU에서 이미지를 추론합니다.

    Args:
        input_tensor (numpy.ndarray): 전처리된 이미지
        npu (str): NPU ID ("npu0" 또는 "npu1")

    Returns:
        dict: 추론 결과
    """
    try:
        async with create_queue(MODEL_PATH, worker_num=1, device=npu) as (submitter, receiver):
            # 입력 데이터 제출
            await submitter.submit(input_tensor)

            # 결과 수신
            async for _, outputs in receiver:
                probabilities = softmax(outputs[1][0])
                return {
                    "deepfake_probability": float(probabilities[1]),
                    "normal_probability": float(probabilities[0]),
                }
    except Exception as e:
        print(f"Error during inference on {npu}: {e}")  # 디버깅 로그
        raise


# Softmax 계산 함수
def softmax(logits):
    exp_logits = np.exp(logits) 
    return exp_logits / np.sum(exp_logits)
