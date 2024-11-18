from furiosa.runtime import session



def initialize_model(model_path):
    return {
        "npu0": session.create(model_path, device = "npu0"),
        "npu1": session.create(model_path, device = "npu1")
    }