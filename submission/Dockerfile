ARG BASE_IMAGE=pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime@sha256:0a3b9fedefe1f61ac4d5a9de9015c0863db27ca0fde2d4e37e6268147980b726
FROM ${BASE_IMAGE}
WORKDIR /workspace
COPY . /workspace
RUN pip install --no-cache-dir --no-index --find-links=/workspace/wheels numpy==1.26.4 h5py==3.11.0 scipy==1.14.1 einops==0.8.0 ninja==1.11.1.4
RUN chmod +x /workspace/runtime/track2/vendor/tinycc/bin/tcc /workspace/runtime/track2/vendor/tinycc/cc
ENV OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
