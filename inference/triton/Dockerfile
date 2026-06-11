# Triton Inference Server + the Python deps the Python-backend models need to run
# seq2seq generation through Optimum on ONNX Runtime (CPU).
#
# Bump TRITON_VERSION to a tag that exists on nvcr.io/nvidia/tritonserver.
ARG TRITON_VERSION=24.12-py3
FROM nvcr.io/nvidia/tritonserver:${TRITON_VERSION}

# CUDA-enabled PyTorch is required for Optimum IO binding (output buffer allocation on GPU).
# cu124 wheels are compatible with the CUDA 12.6 runtime in the Triton 24.12 base image.
# sentencepiece + sacremoses are required by the Marian (OPUS-MT) tokenizers.
RUN python3 -m pip install --no-cache-dir \
      torch --index-url https://download.pytorch.org/whl/cu124 \
 && python3 -m pip install --no-cache-dir \
      "transformers>=4.40,<5" \
      "optimum[onnxruntime-gpu]>=1.20" \
      "sentencepiece>=0.1.99" \
      "sacremoses>=0.1.1"

WORKDIR /workspace
