import numpy as np
import onnxruntime as ort
import time

class DIPRunner:
    def __init__(
        self,
        DIP_path: str,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    ):
        self.DIP = ort.InferenceSession(DIP_path, providers=providers)
        
        # Check for invalid models.
        # lightglue_inputs = [i.name for i in self.DIP.get_inputs()]

    def run(self, image1: np.ndarray, image2: np.ndarray):
        input_names = self.DIP.get_inputs()
        print(input_names)
        start_time = time.time()
        flow_up = self.DIP.run(None,
                {
                    "image1": image1,
                    "image2": image2,
                },)
        end_time = time.time()
        print("device:", ort.get_device())
        print("matching time: {:.2f}ms".format((end_time - start_time)*1000))
        return flow_up


