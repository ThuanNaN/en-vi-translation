"""
Stress test for translation endpoints.

Triton backend (direct, default):
    pip install -e '.[worker]'   # tritonclient + numpy
    python scripts/stresstest.py --url localhost:8000 --requests 200 --concurrency 16
    python scripts/stresstest.py --direction en-vi --output report.txt

Async Triton benchmark (higher GPU utilisation):
    python scripts/stresstest.py --async --requests 500 --concurrency 32 --text-size long
    python scripts/stresstest.py --async --direction en-vi --requests 200 --concurrency 16

vLLM backend (direct OpenAI-compatible API):
    python scripts/stresstest.py --target vllm --vllm-url localhost:8010
    python scripts/stresstest.py --target vllm --vllm-url localhost:8010 --direction en-vi --text-size medium

FastAPI app backend — NMT (Triton):
    python scripts/stresstest.py --target app --api-url http://localhost:80 --api-key changeme
    python scripts/stresstest.py --target app --api-key changeme --direction en-vi

FastAPI app backend — LLM (vLLM):
    python scripts/stresstest.py --target app --api-key changeme --model llm
    python scripts/stresstest.py --target app --api-key changeme --model llm --direction vi-en

Save results to a JSON file:
    python scripts/stresstest.py --results-file results/stress_en_vi.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

_SAMPLE_TEXTS: dict[str, list[str]] = {
    "translator_en_vi": [
        "Hello, how are you?",
        "The weather is nice today.",
        "I would like to order a coffee, please.",
        "Where is the nearest hospital?",
        "Thank you very much for your help.",
        "Can you speak more slowly?",
        "What time does the train depart?",
        "I need to find a pharmacy.",
        "The food here is delicious.",
        "How much does this cost?",
    ],
    "translator_vi_en": [
        "Xin chào, bạn có khỏe không?",
        "Thời tiết hôm nay rất đẹp.",
        "Tôi muốn đặt một ly cà phê.",
        "Bệnh viện gần nhất ở đâu?",
        "Cảm ơn bạn rất nhiều vì đã giúp đỡ.",
        "Bạn có thể nói chậm hơn không?",
        "Tàu khởi hành lúc mấy giờ?",
        "Tôi cần tìm một hiệu thuốc.",
        "Món ăn ở đây rất ngon.",
        "Cái này giá bao nhiêu?",
    ],
}

# Multi-sentence inputs: each generates ~40-80 tokens — enough to keep the GPU busy per
# request so that concurrent batches actually saturate CUDA cores.
_MEDIUM_TEXTS: dict[str, list[str]] = {
    "translator_en_vi": [
        (
            "The development of artificial intelligence has transformed many industries in "
            "recent years. Companies are investing heavily in machine learning research and "
            "infrastructure. These technologies can help automate complex tasks and improve "
            "overall efficiency. However, there are also growing concerns about job "
            "displacement and data privacy."
        ),
        (
            "Vietnam is a beautiful country located in Southeast Asia with a rich history "
            "and vibrant culture. The country stretches over one thousand kilometres from "
            "north to south. It has diverse landscapes including mountains, rivers, and a "
            "long coastline along the South China Sea. Vietnamese cuisine is renowned for "
            "its fresh ingredients, aromatic herbs, and bold flavours."
        ),
        (
            "Climate change is one of the most pressing challenges facing humanity today. "
            "Rising global temperatures are causing more frequent and severe weather events. "
            "Governments and scientists around the world are working together to develop "
            "renewable energy solutions and reduce carbon emissions. Individual actions, "
            "such as reducing energy consumption and choosing sustainable products, also "
            "play an important role."
        ),
        (
            "Education is the foundation of a prosperous society. Access to quality "
            "education empowers individuals to reach their full potential and contribute "
            "meaningfully to their communities. Modern technology has opened new avenues "
            "for learning, allowing students to access educational resources from anywhere "
            "in the world. Online platforms and digital tools are transforming how knowledge "
            "is shared and acquired."
        ),
        (
            "The global economy has undergone significant changes over the past decade. "
            "International trade and investment flows have increased dramatically due to "
            "advances in transportation and communication technology. Emerging markets in "
            "Asia, Africa, and Latin America are playing an increasingly important role in "
            "the world economy. Businesses must adapt to rapidly changing market conditions "
            "in order to remain competitive."
        ),
    ],
    "translator_vi_en": [
        (
            "Trí tuệ nhân tạo đang thay đổi nhiều ngành công nghiệp trong những năm gần "
            "đây. Các công ty đang đầu tư mạnh mẽ vào nghiên cứu và cơ sở hạ tầng học "
            "máy. Những công nghệ này có thể giúp tự động hóa các tác vụ phức tạp và cải "
            "thiện hiệu quả tổng thể. Tuy nhiên, cũng có những lo ngại ngày càng tăng về "
            "việc mất việc làm và quyền riêng tư dữ liệu."
        ),
        (
            "Việt Nam là một đất nước tươi đẹp nằm ở Đông Nam Á với lịch sử phong phú và "
            "văn hóa sôi động. Đất nước trải dài hơn một nghìn kilômét từ bắc vào nam. "
            "Việt Nam có địa hình đa dạng bao gồm núi non, sông ngòi và đường bờ biển dài "
            "dọc theo Biển Đông. Ẩm thực Việt Nam nổi tiếng với nguyên liệu tươi, rau thơm "
            "và hương vị đậm đà."
        ),
        (
            "Biến đổi khí hậu là một trong những thách thức cấp bách nhất mà nhân loại "
            "đang phải đối mặt. Nhiệt độ toàn cầu tăng cao đang gây ra các hiện tượng thời "
            "tiết ngày càng khắc nghiệt và thường xuyên hơn. Các chính phủ và nhà khoa học "
            "trên toàn thế giới đang hợp tác để phát triển các giải pháp năng lượng tái tạo "
            "và giảm lượng khí thải carbon."
        ),
        (
            "Giáo dục là nền tảng của một xã hội phồn thịnh. Tiếp cận giáo dục chất lượng "
            "trao quyền cho cá nhân phát huy hết tiềm năng của mình và đóng góp có ý nghĩa "
            "cho cộng đồng. Công nghệ hiện đại đã mở ra những con đường học tập mới, cho "
            "phép học sinh tiếp cận tài nguyên giáo dục từ bất kỳ đâu trên thế giới."
        ),
        (
            "Nền kinh tế toàn cầu đã trải qua những thay đổi đáng kể trong thập kỷ qua. "
            "Thương mại và đầu tư quốc tế đã tăng trưởng mạnh mẽ nhờ những tiến bộ trong "
            "công nghệ vận tải và truyền thông. Các thị trường mới nổi ở châu Á, châu Phi "
            "và châu Mỹ Latinh đang đóng vai trò ngày càng quan trọng trong nền kinh tế "
            "thế giới."
        ),
    ],
}

# Paragraph-length inputs: each generates ~120-200 tokens — maximises GPU compute time
# per request and makes dynamic-batching saturation very visible.
_LONG_TEXTS: dict[str, list[str]] = {
    "translator_en_vi": [
        (
            "Artificial intelligence and machine learning have become transformative forces "
            "in the modern world. Over the past decade, breakthroughs in deep learning have "
            "enabled computers to perform tasks that were once thought to be exclusively "
            "within the domain of human intelligence, such as recognising images, "
            "understanding speech, and translating languages. Large language models trained "
            "on vast amounts of text data can now generate coherent and contextually "
            "appropriate responses to a wide range of questions. These advances are driving "
            "innovation across industries, from healthcare and finance to transportation and "
            "entertainment. However, the rapid pace of AI development also raises important "
            "ethical questions about bias, transparency, and accountability. Researchers, "
            "policymakers, and businesses must work together to ensure that AI systems are "
            "developed and deployed responsibly, in ways that benefit society as a whole "
            "while minimising potential harms."
        ),
        (
            "Vietnam's economic transformation over the past three decades is one of the "
            "most remarkable development stories in modern history. Since the introduction "
            "of the Doi Moi reforms in 1986, the country has shifted from a centrally "
            "planned economy to a market-oriented one, resulting in rapid GDP growth and a "
            "dramatic reduction in poverty. Millions of Vietnamese citizens have been lifted "
            "out of poverty, and the country has become an important hub for manufacturing "
            "and export in Southeast Asia. Foreign direct investment has poured in as "
            "multinational companies seek to diversify their supply chains away from China. "
            "At the same time, Vietnam is investing heavily in education, technology, and "
            "infrastructure to move up the value chain and develop a more knowledge-based "
            "economy. The country faces ongoing challenges including environmental "
            "degradation, rising inequality, and the need to strengthen governance "
            "institutions, but its prospects for continued growth remain strong."
        ),
        (
            "The transition to renewable energy is one of the defining challenges of the "
            "twenty-first century. Fossil fuels, which have powered economic growth for "
            "over a century, are the primary driver of climate change. Solar and wind power "
            "have become dramatically cheaper in recent years, making them increasingly "
            "competitive with coal and natural gas. Battery storage technology is improving "
            "rapidly, helping to address the intermittency problem that has historically "
            "limited the reliability of renewable sources. Electric vehicles are gaining "
            "market share, reducing dependence on petroleum in the transportation sector. "
            "Governments around the world are setting ambitious targets for carbon "
            "neutrality and investing in green infrastructure. However, the transition "
            "requires massive capital investment, significant changes to electricity grid "
            "architecture, and careful management of the social impacts on communities "
            "that depend on fossil fuel industries for their livelihoods."
        ),
    ],
    "translator_vi_en": [
        (
            "Trí tuệ nhân tạo và học máy đã trở thành những lực lượng chuyển đổi trong "
            "thế giới hiện đại. Trong thập kỷ qua, những đột phá trong học sâu đã cho "
            "phép máy tính thực hiện các nhiệm vụ từng được cho là thuộc về lĩnh vực trí "
            "tuệ con người, chẳng hạn như nhận dạng hình ảnh, hiểu giọng nói và dịch "
            "ngôn ngữ. Các mô hình ngôn ngữ lớn được đào tạo trên lượng dữ liệu văn bản "
            "khổng lồ hiện có thể tạo ra các phản hồi mạch lạc và phù hợp với ngữ cảnh "
            "cho nhiều loại câu hỏi. Những tiến bộ này đang thúc đẩy đổi mới trong các "
            "ngành công nghiệp, từ y tế và tài chính đến vận tải và giải trí. Tuy nhiên, "
            "tốc độ phát triển nhanh chóng của AI cũng đặt ra những câu hỏi đạo đức quan "
            "trọng về sự thiên vị, tính minh bạch và trách nhiệm giải trình. Các nhà "
            "nghiên cứu, nhà hoạch định chính sách và doanh nghiệp phải cùng nhau đảm bảo "
            "rằng các hệ thống AI được phát triển và triển khai có trách nhiệm."
        ),
        (
            "Sự chuyển đổi kinh tế của Việt Nam trong ba thập kỷ qua là một trong những "
            "câu chuyện phát triển đáng chú ý nhất trong lịch sử hiện đại. Kể từ khi "
            "thực hiện cải cách Đổi Mới năm 1986, đất nước đã chuyển từ nền kinh tế kế "
            "hoạch tập trung sang nền kinh tế định hướng thị trường, dẫn đến tăng trưởng "
            "GDP nhanh chóng và giảm nghèo đói đáng kể. Hàng triệu người dân Việt Nam đã "
            "thoát khỏi đói nghèo và đất nước đã trở thành một trung tâm quan trọng về "
            "sản xuất và xuất khẩu ở Đông Nam Á. Đầu tư trực tiếp nước ngoài đổ vào khi "
            "các công ty đa quốc gia tìm cách đa dạng hóa chuỗi cung ứng của họ. Đồng "
            "thời, Việt Nam đang đầu tư mạnh mẽ vào giáo dục, công nghệ và cơ sở hạ tầng "
            "để tiến lên chuỗi giá trị và phát triển nền kinh tế dựa trên tri thức hơn."
        ),
        (
            "Quá trình chuyển đổi sang năng lượng tái tạo là một trong những thách thức "
            "định nghĩa của thế kỷ hai mươi mốt. Nhiên liệu hóa thạch, đã thúc đẩy tăng "
            "trưởng kinh tế trong hơn một thế kỷ, là nguyên nhân chính gây ra biến đổi "
            "khí hậu. Năng lượng mặt trời và gió đã trở nên rẻ hơn đáng kể trong những "
            "năm gần đây, khiến chúng ngày càng cạnh tranh với than đá và khí đốt tự "
            "nhiên. Công nghệ lưu trữ pin đang cải thiện nhanh chóng, giúp giải quyết vấn "
            "đề gián đoạn đã từng hạn chế độ tin cậy của các nguồn tái tạo. Xe điện đang "
            "chiếm thị phần ngày càng lớn, giảm sự phụ thuộc vào dầu mỏ trong lĩnh vực "
            "giao thông. Các chính phủ trên thế giới đang đặt ra các mục tiêu đầy tham "
            "vọng về trung hòa carbon và đầu tư vào cơ sở hạ tầng xanh."
        ),
    ],
}

_TEXT_POOLS: dict[str, dict[str, list[str]]] = {
    "short": _SAMPLE_TEXTS,
    "medium": _MEDIUM_TEXTS,
    "long": _LONG_TEXTS,
}

_MODEL_TO_DIRECTION: dict[str, str] = {
    "translator_en_vi": "en-vi",
    "translator_vi_en": "vi-en",
}

# vLLM: maps direction key → (src language name, tgt language name) for the prompt
_VLLM_DIRECTION_LANGS: dict[str, tuple[str, str]] = {
    "en-vi": ("English", "Vietnamese"),
    "vi-en": ("Vietnamese", "English"),
}

# Maps direction key → text-pool key (shared between Triton and vLLM)
_DIRECTION_TO_POOL_KEY: dict[str, str] = {
    "en-vi": "translator_en_vi",
    "vi-en": "translator_vi_en",
}

MODELS = list(_SAMPLE_TEXTS.keys())


# ---------------------------------------------------------------------------
# Triton backend
# ---------------------------------------------------------------------------

_thread_local = threading.local()


def _get_triton_client(url: str):
    if not hasattr(_thread_local, "client"):
        import tritonclient.http as httpclient  # type: ignore[import-untyped]

        _thread_local.client = httpclient.InferenceServerClient(url=url)

    return _thread_local.client


def _translate_one_triton(url: str, model_name: str, text: str) -> str:
    import numpy as np
    import tritonclient.http as httpclient  # type: ignore[import-untyped]

    client = _get_triton_client(url)

    inp = httpclient.InferInput("INPUT_TEXT", [1, 1], "BYTES")
    inp.set_data_from_numpy(np.array([[text]], dtype=object))

    out = httpclient.InferRequestedOutput("OUTPUT_TEXT")

    result = client.infer(model_name=model_name, inputs=[inp], outputs=[out])
    value = result.as_numpy("OUTPUT_TEXT").reshape(-1)[0]

    return value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else str(value)


def _make_triton_translate_fn(url: str, model_name: str) -> Callable[[str], str]:
    def fn(text: str) -> str:
        return _translate_one_triton(url, model_name, text)

    return fn


def run_stress_async(
    url: str,
    model_name: str,
    texts: list[str],
    n_requests: int,
    concurrency: int,
) -> dict:
    """
    Async Triton benchmark via tritonclient.http.aio (asyncio HTTP).

    Uses asyncio.Semaphore to keep exactly `concurrency` requests in-flight at
    once without blocking OS threads. This saturates Triton's dynamic batcher
    with far more queued requests than the thread-pool approach allows, which
    drives larger batches and higher GPU utilisation.
    """
    import asyncio
    import numpy as np
    import tritonclient.http.aio as httpclient_aio  # type: ignore[import-untyped]

    latencies: list[float] = [0.0] * n_requests
    error_list: list[str] = []

    async def _one_request(client, idx: int, sem: asyncio.Semaphore) -> None:
        text = texts[idx % len(texts)]
        inp = httpclient_aio.InferInput("INPUT_TEXT", [1, 1], "BYTES")
        inp.set_data_from_numpy(np.array([[text]], dtype=object))
        out = httpclient_aio.InferRequestedOutput("OUTPUT_TEXT")
        async with sem:
            t0 = time.perf_counter()
            try:
                await client.infer(model_name=model_name, inputs=[inp], outputs=[out])
                latencies[idx] = time.perf_counter() - t0
            except Exception as exc:
                latencies[idx] = time.perf_counter() - t0
                error_list.append(str(exc))

    async def _run() -> float:
        sem = asyncio.Semaphore(concurrency)
        async with httpclient_aio.InferenceServerClient(url=url) as client:
            t_start = time.perf_counter()
            await asyncio.gather(*[_one_request(client, i, sem) for i in range(n_requests)])
            return time.perf_counter() - t_start

    total_seconds = asyncio.run(_run())

    return {
        "mode": "stress",
        "model": model_name,
        "n_requests": n_requests,
        "concurrency": concurrency,
        "total_seconds": total_seconds,
        "latencies": latencies,
        "errors": error_list,
    }


# ---------------------------------------------------------------------------
# vLLM backend (direct OpenAI-compatible chat API)
# ---------------------------------------------------------------------------

def _translate_one_vllm(url: str, model_name: str, direction: str, text: str) -> str:
    src_lang, tgt_lang = _VLLM_DIRECTION_LANGS[direction]
    payload = json.dumps({
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    f"Translate from {src_lang} to {tgt_lang}. "
                    "Output only the translation, no explanations."
                ),
            },
            {"role": "user", "content": text},
        ],
        "max_tokens": 512,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        f"http://{url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"].strip()


def _make_vllm_translate_fn(url: str, model_name: str, direction: str) -> Callable[[str], str]:
    def fn(text: str) -> str:
        return _translate_one_vllm(url, model_name, direction, text)
    return fn


def _check_vllm_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"http://{url}/health", timeout=5) as resp:
            return 200 <= resp.status < 500
    except Exception:
        return False


# ---------------------------------------------------------------------------
# App backend: FastAPI POST /translate -> poll GET /jobs/{job_id}
# ---------------------------------------------------------------------------

def _translate_one_app(
    api_url: str,
    api_key: str,
    direction: str,
    text: str,
    poll_interval: float = 0.5,
    timeout: float = 60.0,
    model: str | None = None,
) -> str:
    api_url = api_url.rstrip("/")

    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key,
    }

    body: dict = {"text": text, "direction": direction}
    if model:
        body["model"] = model

    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(
        f"{api_url}/translate",
        data=payload,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=min(timeout, 10.0)) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            job_id = data["job_id"]
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST /translate failed {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"POST /translate failed: {exc}") from exc

    poll_url = f"{api_url}/jobs/{job_id}"
    poll_headers = {"X-API-Key": api_key}
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        req = urllib.request.Request(poll_url, headers=poll_headers)

        try:
            with urllib.request.urlopen(req, timeout=min(timeout, 10.0)) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GET /jobs/{job_id} failed {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"GET /jobs/{job_id} failed: {exc}") from exc

        status = data.get("status")

        if status == "done":
            return str(data.get("translation", ""))

        if status == "failed":
            raise RuntimeError(f"Translation failed: {data.get('error')}")

        time.sleep(poll_interval)

    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


def _make_app_translate_fn(
    api_url: str,
    api_key: str,
    direction: str,
    poll_interval: float,
    timeout: float,
    model: str | None = None,
) -> Callable[[str], str]:
    def fn(text: str) -> str:
        return _translate_one_app(
            api_url=api_url,
            api_key=api_key,
            direction=direction,
            text=text,
            poll_interval=poll_interval,
            timeout=timeout,
            model=model,
        )

    return fn


def _check_app_ready(api_url: str) -> bool:
    api_url = api_url.rstrip("/")

    for path in ["/health", "/ready", "/docs", "/"]:
        try:
            with urllib.request.urlopen(f"{api_url}{path}", timeout=5) as resp:
                if 200 <= resp.status < 500:
                    return True
        except Exception:
            continue

    return False


# ---------------------------------------------------------------------------
# Generic stress / eval runners
# ---------------------------------------------------------------------------

def _worker(
    translate_fn: Callable[[str], str],
    texts: list[str],
    idx: int,
) -> tuple[float, str, str | None]:
    text = texts[idx % len(texts)]
    t0 = time.perf_counter()

    try:
        hypothesis = translate_fn(text)
        return time.perf_counter() - t0, hypothesis, None
    except Exception as exc:
        return time.perf_counter() - t0, "", str(exc)


def run_stress(
    translate_fn: Callable[[str], str],
    model_name: str,
    n_requests: int,
    concurrency: int,
    texts: list[str] | None = None,
    collect_translations: bool = False,
) -> dict:
    if texts is None:
        texts = _SAMPLE_TEXTS[model_name]

    latencies: list[float] = []
    errors: list[str] = []
    per_item: list[dict] = []

    t_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(_worker, translate_fn, texts, i): i
            for i in range(n_requests)
        }

        for fut in as_completed(futures):
            i = futures[fut]
            latency, hyp, err = fut.result()

            latencies.append(latency)

            if err:
                errors.append(err)

            if collect_translations:
                entry: dict = {
                    "index": i,
                    "source": texts[i % len(texts)],
                    "hypothesis": hyp,
                    "latency_s": round(latency, 4),
                }

                if err:
                    entry["error"] = err

                per_item.append(entry)

    total_seconds = time.perf_counter() - t_start

    result: dict = {
        "mode": "stress",
        "model": model_name,
        "n_requests": n_requests,
        "concurrency": concurrency,
        "total_seconds": total_seconds,
        "latencies": latencies,
        "errors": errors,
    }

    if collect_translations:
        result["translations"] = sorted(per_item, key=lambda x: x["index"])

    return result


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0

    s = sorted(data)
    k = (len(s) - 1) * p / 100

    lo = int(k)
    hi = min(lo + 1, len(s) - 1)

    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def build_report(results: list[dict]) -> str:
    width = 64
    lines: list[str] = []

    lines.append("=" * width)
    lines.append("  STRESS TEST REPORT")
    lines.append("=" * width)

    for r in results:
        lines.append("")
        _append_stress_section(lines, r, width)

    return "\n".join(lines)


def _append_stress_section(lines: list[str], r: dict, width: int) -> None:
    n = r["n_requests"]
    n_err = len(r["errors"])
    n_ok = n - n_err
    lats = r["latencies"]

    lines.append(f"Model        : {r['model']}")

    if r.get("backend"):
        lines.append(f"Backend      : {r['backend']}")

    lines.append(f"Requests     : {n}  (concurrency={r['concurrency']})")
    lines.append(f"Succeeded    : {n_ok}   Failed: {n_err}")
    lines.append(f"Duration     : {r['total_seconds']:.2f} s")

    if r["total_seconds"] > 0:
        lines.append(f"Throughput   : {n / r['total_seconds']:.2f} req/s")

    if lats:
        lines.append("Latency (s)  :")
        lines.append(f"  min  {min(lats):.3f}   mean {statistics.mean(lats):.3f}")
        lines.append(
            f"  p50  {_percentile(lats, 50):.3f}"
            f"   p95  {_percentile(lats, 95):.3f}"
            f"   p99  {_percentile(lats, 99):.3f}"
            f"   max  {max(lats):.3f}"
        )

    if r["errors"]:
        lines.append(f"First error  : {r['errors'][0]}")

    lines.append("-" * width)


# ---------------------------------------------------------------------------
# JSON results writer
# ---------------------------------------------------------------------------

def _latency_stats(latencies: list[float]) -> dict:
    return {
        "min_s": round(min(latencies), 4),
        "mean_s": round(statistics.mean(latencies), 4),
        "p50_s": round(_percentile(latencies, 50), 4),
        "p95_s": round(_percentile(latencies, 95), 4),
        "p99_s": round(_percentile(latencies, 99), 4),
        "max_s": round(max(latencies), 4),
    }


def _write_results_json(results: list[dict], path: Path) -> None:
    """
    Serialize full results to JSON.

    The raw top-level latency list is replaced by summary stats to avoid
    making the file too large. Per-translation latencies are still preserved
    inside the "translations" field when --results-file is used.
    """
    json_results: list[dict] = []

    for r in results:
        latencies = r.get("latencies", [])

        entry = {
            k: v
            for k, v in r.items()
            if k != "latencies"
        }

        if latencies:
            entry["latency_stats"] = _latency_stats(latencies)

        json_results.append(entry)

    output = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "results": json_results,
    }

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--target",
        choices=["triton", "vllm", "app"],
        default="triton",
        help=(
            "Backend to test: 'triton' (direct Triton HTTP), "
            "'vllm' (direct vLLM OpenAI API), "
            "or 'app' (FastAPI POST /translate). Default: triton"
        ),
    )

    triton_grp = parser.add_argument_group("triton backend")
    triton_grp.add_argument(
        "--url",
        default="localhost:8000",
        help="Triton HTTP endpoint. Default: localhost:8000",
    )

    vllm_grp = parser.add_argument_group("vllm backend (--target vllm)")
    vllm_grp.add_argument(
        "--vllm-url",
        default="localhost:8010",
        help="vLLM OpenAI-compatible API endpoint. Default: localhost:8010",
    )
    vllm_grp.add_argument(
        "--vllm-model-name",
        default="Qwen/Qwen3.5-0.8B",
        help="Model name served by vLLM. Default: Qwen/Qwen3.5-0.8B",
    )

    app_grp = parser.add_argument_group("app backend (--target app)")
    app_grp.add_argument(
        "--api-url",
        default="http://localhost:80",
        help="FastAPI base URL. Default: http://localhost:80",
    )
    app_grp.add_argument(
        "--api-key",
        default="changeme",
        help="X-API-Key header value",
    )
    app_grp.add_argument(
        "--model",
        default=None,
        metavar="BACKEND",
        help=(
            "Backend selector for app target: omit for NMT/Triton, "
            "'llm' for vLLM. Passed as 'model' in the request body."
        ),
    )
    app_grp.add_argument(
        "--poll-interval",
        type=float,
        default=0.5,
        help="Seconds between job-status polls in app mode. Default: 0.5",
    )
    app_grp.add_argument(
        "--job-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for a single job to complete in app mode. Default: 60",
    )

    parser.add_argument(
        "--requests",
        type=int,
        default=100,
        help="Total requests per model in stress mode",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Number of concurrent workers (or in-flight requests in --async mode)",
    )
    parser.add_argument(
        "--text-size",
        choices=["short", "medium", "long"],
        default="short",
        help=(
            "Size of synthetic inputs: short (~5 tokens), medium (~60 tokens), "
            "long (~150 tokens). Longer texts generate more GPU work per request "
            "and are better for measuring peak throughput. Default: short"
        ),
    )
    parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        default=False,
        help=(
            "Use tritonclient async_infer instead of a thread pool. "
            "Fires all requests as fast as possible so Triton's dynamic batcher "
            "sees many in-flight requests simultaneously — recommended for GPU "
            "saturation benchmarks. Only valid for --target triton."
        ),
    )
    parser.add_argument(
        "--direction",
        choices=["en-vi", "vi-en", "both"],
        default="both",
        help="Which translation direction to test. Default: both",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        help="Also write the text report to this file",
    )
    parser.add_argument(
        "--results-file",
        metavar="JSON_FILE",
        help=(
            "Save all translation results to this JSON file. "
            "Includes source, hypothesis, reference when available, latency, and errors."
        ),
    )

    args = parser.parse_args()

    if args.concurrency <= 0:
        raise ValueError("--concurrency must be greater than 0")

    if args.requests <= 0:
        raise ValueError("--requests must be greater than 0")

    if args.use_async and args.target != "triton":
        print("--async is only supported with --target triton", file=sys.stderr)
        raise SystemExit(1)

    if args.model and args.target != "app":
        print("--model is only used with --target app", file=sys.stderr)
        raise SystemExit(1)

    args.api_url = args.api_url.rstrip("/")

    collect_translations = bool(args.results_file)
    text_pool = _TEXT_POOLS[args.text_size]

    if args.target == "triton":
        try:
            import tritonclient.http as httpclient  # type: ignore[import-untyped]
        except ImportError:
            print(
                "tritonclient not found. Install with: pip install -e '.[worker]'",
                file=sys.stderr,
            )
            raise SystemExit(1)

        client = httpclient.InferenceServerClient(url=args.url)

        if not client.is_server_ready():
            print(f"Triton at {args.url} is not ready.", file=sys.stderr)
            raise SystemExit(1)

        model_map = {
            "en-vi": ["translator_en_vi"],
            "vi-en": ["translator_vi_en"],
            "both": MODELS,
        }
        selected = model_map[args.direction]

        def _is_ready(model_name: str) -> bool:
            return client.is_model_ready(model_name)

        def _make_fn(model_name: str) -> Callable[[str], str]:
            return _make_triton_translate_fn(args.url, model_name)

        backend_label = f"triton ({args.url})"

    elif args.target == "vllm":
        if not _check_vllm_ready(args.vllm_url):
            print(f"vLLM at {args.vllm_url} is not reachable.", file=sys.stderr)
            raise SystemExit(1)

        direction_map = {
            "en-vi": ["en-vi"],
            "vi-en": ["vi-en"],
            "both": list(_VLLM_DIRECTION_LANGS.keys()),
        }
        selected = direction_map[args.direction]

        def _is_ready(_: str) -> bool:
            return True

        def _make_fn(direction: str) -> Callable[[str], str]:  # type: ignore[misc]
            return _make_vllm_translate_fn(args.vllm_url, args.vllm_model_name, direction)

        backend_label = f"vllm ({args.vllm_url}, {args.vllm_model_name})"

    else:  # app
        if not _check_app_ready(args.api_url):
            print(f"FastAPI app at {args.api_url} is not reachable.", file=sys.stderr)
            raise SystemExit(1)

        model_map = {
            "en-vi": ["translator_en_vi"],
            "vi-en": ["translator_vi_en"],
            "both": MODELS,
        }
        selected = model_map[args.direction]

        def _is_ready(_model_name: str) -> bool:
            return True

        def _make_fn(model_name: str) -> Callable[[str], str]:
            direction = _MODEL_TO_DIRECTION[model_name]
            return _make_app_translate_fn(
                api_url=args.api_url,
                api_key=args.api_key,
                direction=direction,
                poll_interval=args.poll_interval,
                timeout=args.job_timeout,
                model=args.model,
            )

        backend_label = f"app ({args.api_url})" + (f" model={args.model}" if args.model else "")

    results: list[dict] = []

    for model_name in selected:
        if not _is_ready(model_name):
            print(f"[skip] {model_name} not ready", file=sys.stderr)
            continue

        pool_key = _DIRECTION_TO_POOL_KEY.get(model_name, model_name)
        texts = text_pool[pool_key]

        if args.use_async:
            mode_label = "async"
            print(
                f"[{mode_label}] {model_name} ({backend_label}): "
                f"{args.requests} requests, concurrency={args.concurrency}, "
                f"text-size={args.text_size}, samples={len(texts)}",
                file=sys.stderr,
                flush=True,
            )
            result = run_stress_async(
                url=args.url,
                model_name=model_name,
                texts=texts,
                n_requests=args.requests,
                concurrency=args.concurrency,
            )
        else:
            print(
                f"[stress] {model_name} ({backend_label}): "
                f"{args.requests} requests, concurrency={args.concurrency}, "
                f"text-size={args.text_size}, samples={len(texts)}",
                file=sys.stderr,
                flush=True,
            )
            translate_fn = _make_fn(model_name)
            result = run_stress(
                translate_fn=translate_fn,
                model_name=model_name,
                n_requests=args.requests,
                concurrency=args.concurrency,
                texts=texts,
                collect_translations=collect_translations,
            )

        result["backend"] = backend_label
        results.append(result)

    if not results:
        print("No models available.", file=sys.stderr)
        raise SystemExit(1)

    report = build_report(results)
    print(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(report + "\n")

        print(f"\nReport saved to {args.output}", file=sys.stderr)

    if args.results_file:
        results_path = Path(args.results_file)
        _write_results_json(results, results_path)

        n_translations = sum(len(r.get("translations", [])) for r in results)

        print(
            f"\nResults ({n_translations} translations) saved to {results_path}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()