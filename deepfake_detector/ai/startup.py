from furiosa.runtime import create_queue
import asyncio

# 전역 변수로 큐 관리
queues = {"npu0": None, "npu1": None}

async def initialize_queues(model_path, num_worker=1):
    """
    NPU별로 큐를 초기화합니다.
    Args:
        model_path (str): ONNX 모델 경로
        num_worker (int): 워커 수
    """
    global queues

    queues["npu0"] = await create_queue(model_path, worker_num=num_worker, device="npu0")
    queues["npu1"] = await create_queue(model_path, worker_num=num_worker, device="npu1")

    print("Queues initialized!")
