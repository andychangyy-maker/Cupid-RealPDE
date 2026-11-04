"""Capture input/output transfers with the existing fixed CUDA inference graph."""
import numpy as np
import torch


class CapturedIO:
    def __init__(self,operation,batch,device):
        shape=(batch,20,32,64,3)
        self.input=torch.zeros(shape,dtype=torch.float32,pin_memory=True)
        self.output=torch.empty((3,*shape),dtype=torch.float32,pin_memory=True)
        self.device_input=torch.empty(shape,dtype=torch.float32,device=device)
        self.device=device
        stream=torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream),torch.inference_mode():
            for _ in range(3):
                self.device_input.copy_(self.input,non_blocking=True)
                self.device_output=operation(self.device_input)
                self.output.copy_(self.device_output,non_blocking=True)
        torch.cuda.current_stream(device).wait_stream(stream);torch.cuda.synchronize(device)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream),torch.inference_mode():
            self.device_input.copy_(self.input,non_blocking=True)
            self.device_output=operation(self.device_input)
            self.output.copy_(self.device_output,non_blocking=True)
        torch.cuda.current_stream(device).wait_stream(stream)
        self.input_numpy=self.input.numpy();self.output_numpy=self.output.numpy()

    def __call__(self,value):
        np.copyto(self.input_numpy,value)
        self.graph.replay();torch.cuda.current_stream(self.device).synchronize()
        # Each public return owns its memory across subsequent graph replays.
        return self.output_numpy.copy()
