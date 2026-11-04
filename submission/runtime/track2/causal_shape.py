"""Causal tangent-weighted shape fusion with an independent teacher stream."""
from pathlib import Path
import warnings

import torch
from shape_model import PointModel
from moment_fused import HistoryMoment


def regional(value):
    return value.reshape(1,20,2,16,4,16,2).sum((0,1,3,5,6)).reshape(8)


class ShapeFusion:
    def __init__(self, streaming, directory):
        self.input=streaming._input
        self.point=streaming._output[0]
        self.target=streaming._target
        self.count=streaming._count
        self.revealed=streaming._revealed
        self.stats=streaming.official_stats
        self.device=streaming.device
        self.moment=HistoryMoment(streaming)
        self.teacher=PointModel().to(device=self.device,dtype=streaming.dtype).eval().requires_grad_(False)
        payload=torch.load(Path(directory)/'shape_model.pth',map_location='cpu',weights_only=False)
        self.teacher.load_state_dict(payload['state_dict'],strict=True)
        self.cross=torch.zeros(8,device=self.device)
        self.square=torch.zeros_like(self.cross)
        self.weights=torch.full_like(self.cross,.5)
        self.previous_center=torch.zeros(1,20,32,64,2,device=self.device)
        self.previous_tangent=torch.zeros_like(self.previous_center)
        self.teacher_graph=self.fusion_graph=None
        self.compiled=False
        self.teacher_operation=self.teacher_forward
        self.operation=self.fuse
        if self.device.type=='cuda':
            for layer in self.teacher.modules():
                if isinstance(layer,torch.nn.Conv2d):
                    layer.to(memory_format=torch.channels_last)
            try:
                self.teacher_operation=torch.compile(self.teacher_forward,fullgraph=True,dynamic=False)
                self.operation=torch.compile(self.fuse,fullgraph=True,dynamic=False)
                with torch.inference_mode():
                    self.teacher_output=self.teacher_operation(self.input)
                    self.operation(self.point,self.teacher_output)
                torch.cuda.synchronize(self.device)
            except Exception as error:
                self.teacher_operation=self.teacher_forward
                self.operation=self.fuse
                warnings.warn('Causal shape compiler unavailable: '+repr(error))
            else:
                self.compiled=True
                print('[track2] compiled causal shape ready',flush=True)
            self.stream=torch.cuda.Stream(device=self.device)
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self.stream),torch.inference_mode():
                for _ in range(3):
                    self.teacher_output=self.teacher_operation(self.input)
            self.teacher_graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.teacher_graph,stream=self.stream),torch.inference_mode():
                self.teacher_output=self.teacher_operation(self.input)
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
            with torch.inference_mode():
                for _ in range(3):
                    self.output=self.operation(self.point,self.teacher_output)
            self.fusion_graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.fusion_graph),torch.inference_mode():
                self.output=self.operation(self.point,self.teacher_output)
        self.reset()

    def reset(self):
        self.cross.zero_();self.square.zero_();self.weights.fill_(.5)
        self.moment.reset()
        self.previous_center.zero_();self.previous_tangent.zero_()

    def teacher_forward(self,value):
        raw=value.unsqueeze(0)*self.stats['std_input']+self.stats['mean_input']
        return self.teacher(raw).float()

    def fuse(self,point,t):
        y=(self.target*self.stats['std_target']+self.stats['mean_target']).unsqueeze(0)[...,:2]
        norm=y.square().sum().clamp_min(1e-12)
        self.cross.add_(regional(self.previous_tangent*(y-self.previous_center))/norm*self.revealed)
        self.square.add_(regional(self.previous_tangent.square())/norm*self.revealed)
        shrink=self.count/(self.count+5)*(self.count>=3)
        self.weights.copy_(.5+shrink*(self.cross/self.square.clamp_min(1e-12)).clamp(-.5,.5))
        p=point.unsqueeze(0)*self.stats['std_target']+self.stats['mean_target']
        mean=p[...,:2].mean(1,keepdim=True)
        pd=p[...,:2]-mean
        td=t-t.mean(1,keepdim=True)
        pe=pd.square().sum(-1,keepdim=True).mean(1,keepdim=True)
        te=td.square().sum(-1,keepdim=True).mean(1,keepdim=True)
        td=td*(pe/te.clamp_min(1e-12)).sqrt()
        mid=.5*(pd+td)
        em=mid.square().sum(-1,keepdim=True).mean(1,keepdim=True)
        valid=(te>1e-12)&(em>pe*1e-6)
        gain=(pe/em.clamp_min(1e-12)).sqrt()
        self.previous_center.copy_(mean+torch.where(valid,mid*gain,pd))
        self.previous_tangent.copy_(torch.where(valid,(td-pd)*gain,torch.zeros_like(pd)))
        w=self.weights.reshape(1,1,2,1,4,1,1)
        mix=(pd.reshape(1,20,2,16,4,16,2)*(1-w)+td.reshape(1,20,2,16,4,16,2)*w).reshape_as(pd)
        energy=mix.square().sum(-1,keepdim=True).mean(1,keepdim=True)
        dynamic=torch.where((te>1e-12)&(energy>pe*1e-6),mix*(pe/energy.clamp_min(1e-12)).sqrt(),pd)
        result=torch.cat((mean+dynamic,p[...,2:]),-1)
        return self.moment.apply((result-self.stats['mean_target'])/self.stats['std_target'])

    def start(self):
        if self.teacher_graph is None:
            self.teacher_output=self.teacher_operation(self.input)
        else:
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self.stream):
                self.teacher_graph.replay()

    def predict(self):
        if self.fusion_graph is None:
            return self.operation(self.point,self.teacher_output)
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        self.fusion_graph.replay()
        return self.output.clone()
