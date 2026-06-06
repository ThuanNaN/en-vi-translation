"""Export the pretrained EN<->VI HuggingFace checkpoints to ONNX for Triton.

For each direction it downloads the HF checkpoint, exports it to ONNX with 🤗 Optimum,
and writes the ONNX graph + tokenizer into::

    <model_repository>/<triton_name>/1/onnx/

which is exactly where model.py (the Triton Python backend) expects to find it.

Run it inside the Triton image, which already has the heavy deps::

    make export
    # equivalently:
    docker compose run --rm --no-deps -w /workspace triton \
        python3 scripts/export_models.py --model-repository /models

Or locally in a Python 3.10-3.12 venv after `pip install -e '.[export]'`.

Model ids default to the Helsinki-NLP OPUS-MT pair and can be overridden with the
ENVIT5_HF_MODEL_EN_VI / ENVIT5_HF_MODEL_VI_EN environment variables.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]

# Triton model name -> (env var, default HuggingFace id)
MODELS: dict[str, tuple[str, str]] = {
    "translator_en_vi": ("ENVIT5_HF_MODEL_EN_VI", "Helsinki-NLP/opus-mt-en-vi"),
    "translator_vi_en": ("ENVIT5_HF_MODEL_VI_EN", "Helsinki-NLP/opus-mt-vi-en"),
}


def export_one(triton_name: str, hf_id: str, repo_dir: Path) -> None:
    out_dir = repo_dir / triton_name / "1" / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[export] {hf_id} -> {out_dir}", flush=True)

    model = ORTModelForSeq2SeqLM.from_pretrained(hf_id, export=True)
    model.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(hf_id).save_pretrained(out_dir)

    print(f"[done]   {triton_name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-repository",
        default=str(REPO_ROOT / "model_repository"),
        help="Path to the Triton model repository (default: ./model_repository).",
    )
    parser.add_argument(
        "--only",
        choices=sorted(MODELS),
        help="Export just one direction instead of both.",
    )
    args = parser.parse_args()

    repo_dir = Path(args.model_repository)
    selected = {args.only: MODELS[args.only]} if args.only else MODELS

    for triton_name, (env_var, default_id) in selected.items():
        hf_id = os.environ.get(env_var, default_id)
        export_one(triton_name, hf_id, repo_dir)


if __name__ == "__main__":
    main()
