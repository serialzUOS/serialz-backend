from furiosa.runtime import create_queue
import asyncio

# 전역 변수로 큐 및 초기화 상태 관리
queues = {"npu0": None, "npu1": None}
is_initialized = False  # 초기화 상태 플래그

async def initialize_queues(model_path, num_worker=4):
    global queues, is_initialized
    print(f"Initializing with model: {model_path}")

    try:
        print("Initializing npu0...")
        queues["npu0"] = await create_queue(model_path, worker_num=num_worker, device="npu0")
        print("npu0 initialized successfully.")
        
        print("Initializing npu1...")
        queues["npu1"] = await create_queue(model_path, worker_num=num_worker, device="npu1")
        print("npu1 initialized successfully.")
        
        is_initialized = queues["npu0"] is not None and queues["npu1"] is not None

        # 디버깅: queues 상태 확인
        print(f"Queues state after initialization: {queues}")

        # 초기화 상태 확인
        is_initialized = queues["npu0"] is not None and queues["npu1"] is not None
        print(f"Initialization state evaluated: {is_initialized}")

        print("Queues initialized!")
    except Exception as e:
        print(f"Failed to initialize queues: {e}")
        is_initialized = False
