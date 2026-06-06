"""Triton Python backend: seq2seq translation via Optimum ONNX Runtime.

Loads the ONNX-exported HuggingFace seq2seq model placed (by scripts/export_models.py)
in the sibling ``onnx/`` directory, then runs autoregressive generation. The model is
string-in / string-out: tokenization and decoding happen here so callers only deal with
plain text.

This file is identical for every direction; the exported ``onnx/`` directory next to it
determines which language pair this model instance translates.
"""

import json
import os

import numpy as np
import triton_python_backend_utils as pb_utils
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        params = model_config.get("parameters", {})

        def _param(name: str, default: str) -> str:
            entry = params.get(name)
            return entry["string_value"] if entry else default

        self.max_new_tokens = int(_param("max_new_tokens", "256"))
        self.num_beams = int(_param("num_beams", "1"))

        onnx_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx")
        if not os.path.isdir(onnx_dir):
            raise pb_utils.TritonModelException(
                f"ONNX directory not found at {onnx_dir}. "
                "Export the model first with `make export` (scripts/export_models.py)."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(onnx_dir)
        self.model = ORTModelForSeq2SeqLM.from_pretrained(onnx_dir)

        output_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT_TEXT")
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

    def execute(self, requests):
        responses = []
        # Triton hands us a list of requests (the dynamic batcher may add several).
        # Each request is translated independently; the strings within one request are
        # batched together in a single generate() call.
        for request in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")
            raw = in_tensor.as_numpy().reshape(-1)
            texts = [
                value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else str(value)
                for value in raw
            ]

            encoded = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True
            )
            generated = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
            )
            translations = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

            out_array = np.array(translations, dtype=object).reshape(-1, 1)
            out_tensor = pb_utils.Tensor("OUTPUT_TEXT", out_array.astype(self.output_dtype))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
        return responses

    def finalize(self):
        self.model = None
        self.tokenizer = None
