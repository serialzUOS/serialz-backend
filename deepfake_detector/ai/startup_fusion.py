from furiosa.runtime import create_queue
import asyncio

# 전역 변수로 큐 및 초기화 상태 관리
fusion_queue = None  # FusionPE 큐
is_initialized = False  # 초기화 상태 플래그

async def initialize_fusion_queue(model_path, num_worker=4):
    global fusion_queue, is_initialized
    print(f"Initializing FusionPE queue with model: {model_path}")

    try:
        # FusionPE 방식으로 단일 큐 생성
        fusion_queue = await create_queue(model_path, worker_num=num_worker, device="npu0,npu1")
        print("FusionPE queue initialized successfully.")
        
        # 초기화 상태 설정
        is_initialized = fusion_queue is not None

        # 디버깅: 초기화 상태 확인
        print(f"Fusion queue state: {fusion_queue}")
        print(f"Initialization state evaluated: {is_initialized}")
    except Exception as e:
        print(f"Failed to initialize FusionPE queue: {e}")
        is_initialized = False
