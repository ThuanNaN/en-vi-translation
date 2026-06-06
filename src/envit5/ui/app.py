"""Gradio demo UI for the EN↔VI translation API."""
from __future__ import annotations

import os
import time
from collections.abc import Generator

import gradio as gr
import requests

API_URL = os.getenv("ENVIT5_API_URL", "http://localhost:8080")
API_KEY = os.getenv("ENVIT5_API_KEY", "changeme")
POLL_INTERVAL = 0.5
TIMEOUT = 60

_DIRECTION_MAP = {
    "Auto-detect": None,
    "EN → VI": "en-vi",
    "VI → EN": "vi-en",
}

_EXAMPLES = [
    ["Hello, how are you today?", "EN → VI"],
    ["The weather is beautiful this morning.", "EN → VI"],
    ["Xin chào, bạn có khỏe không?", "VI → EN"],
    ["Hà Nội là thủ đô của Việt Nam.", "VI → EN"],
    ["I love learning new languages.", "Auto-detect"],
    ["Tôi rất thích học ngôn ngữ mới.", "Auto-detect"],
]


def _headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY, "Content-Type": "application/json"}


def translate(
    text: str, direction: str
) -> Generator[tuple[str, str], None, None]:
    text = text.strip()
    if not text:
        yield "", "Please enter some text."
        return

    yield "", "Submitting..."

    payload: dict = {"text": text}
    dir_value = _DIRECTION_MAP.get(direction)
    if dir_value:
        payload["direction"] = dir_value

    try:
        resp = requests.post(
            f"{API_URL}/translate", json=payload, headers=_headers(), timeout=10
        )
    except requests.ConnectionError:
        yield "", f"Cannot reach API at {API_URL}"
        return

    if resp.status_code != 202:
        yield "", f"API error {resp.status_code}: {resp.text}"
        return

    job_id = resp.json()["job_id"]
    yield "", f"Queued | job: {job_id}"

    deadline = time.monotonic() + TIMEOUT
    while time.monotonic() < deadline:
        time.sleep(POLL_INTERVAL)
        try:
            r = requests.get(
                f"{API_URL}/jobs/{job_id}", headers=_headers(), timeout=10
            )
        except requests.ConnectionError:
            yield "", f"Lost connection while polling | job: {job_id}"
            return

        if r.status_code != 200:
            yield "", f"Poll error {r.status_code} | job: {job_id}"
            return

        data = r.json()
        status = data["status"]

        if status == "done":
            yield data["translation"], f"Done | job: {job_id}"
            return

        if status == "failed":
            yield "", f"Failed: {data.get('error', 'unknown')} | job: {job_id}"
            return

        yield "", f"{status.capitalize()} | job: {job_id}"

    yield "", f"Timeout after {TIMEOUT}s | job: {job_id}"


with gr.Blocks(title="EN↔VI Translation") as demo:
    gr.Markdown(
        """
# EN ↔ VI Neural Machine Translation
Powered by Helsinki-NLP/opus-mt via Triton Inference Server + Celery
"""
    )

    direction = gr.Radio(
        choices=list(_DIRECTION_MAP.keys()),
        value="Auto-detect",
        label="Direction",
    )

    with gr.Row():
        input_box = gr.Textbox(
            label="Source text",
            lines=8,
            placeholder="Type or paste text here…",
        )
        output_box = gr.Textbox(
            label="Translation",
            lines=8,
            interactive=False,
        )

    with gr.Row():
        translate_btn = gr.Button("Translate", variant="primary", scale=1)
        clear_btn = gr.Button("Clear", scale=0)

    status_box = gr.Textbox(label="Status", interactive=False, max_lines=1)

    gr.Examples(
        examples=_EXAMPLES,
        inputs=[input_box, direction],
        label="Examples",
    )

    translate_btn.click(
        fn=translate,
        inputs=[input_box, direction],
        outputs=[output_box, status_box],
    )
    clear_btn.click(
        fn=lambda: ("", "", ""),
        outputs=[input_box, output_box, status_box],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_PORT", "7860")),
        theme=gr.themes.Soft(),
    )
