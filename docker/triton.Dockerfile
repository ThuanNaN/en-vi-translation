# Triton Inference Server + the Python deps the Python-backend models need to run
# seq2seq generation through Optimum on ONNX Runtime (CPU).
#
# Bump TRITON_VERSION to a tag that exists on nvcr.io/nvidia/tritonserver.
ARG TRITON_VERSION=24.12-py3
FROM nvcr.io/nvidia/tritonserver:${TRITON_VERSION}

# CPU-only PyTorch backs transformers' .generate(); ONNX Runtime (via Optimum) does the
# actual encoder/decoder matmuls. sentencepiece + sacremoses are required by the Marian
# (OPUS-MT) tokenizers.
RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir \
      torch --index-url https://download.pytorch.org/whl/cpu \
 && python3 -m pip install --no-cache-dir \
      "transformers>=4.40,<5" \
      "optimum[onnxruntime]>=1.20" \
      "sentencepiece>=0.1.99" \
      "sacremoses>=0.1.1"

WORKDIR /workspace
