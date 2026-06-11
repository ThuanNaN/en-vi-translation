"""
App stress test — exercises the FastAPI app through both inference backends.

Phase 1  NMT via Triton  (fast ONNX seq2seq, higher concurrency)
  • en-vi direction
  • vi-en direction

Phase 2  LLM via vLLM  (autoregressive, lower concurrency)
  • en-vi direction
  • vi-en direction

Quick start:
    python scripts/app_stresstest.py --api-key changeme

Run only one backend:
    python scripts/app_stresstest.py --backend nmt --api-key changeme
    python scripts/app_stresstest.py --backend llm --api-key changeme

Tune load:
    python scripts/app_stresstest.py \\
        --nmt-requests 200 --nmt-concurrency 16 \\
        --llm-requests 50  --llm-concurrency 4  \\
        --text-size medium

Save results to JSON:
    python scripts/app_stresstest.py --api-key changeme \\
        --results-file results/app_stress.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import socket
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Sample text pools (short / medium / long per direction)
# ---------------------------------------------------------------------------

_SHORT: dict[str, list[str]] = {
    "en-vi": [
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
    "vi-en": [
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

_MEDIUM: dict[str, list[str]] = {
    "en-vi": [
        (
            "The development of artificial intelligence has transformed many industries. "
            "Companies are investing heavily in machine learning research and infrastructure. "
            "These technologies can help automate complex tasks and improve overall efficiency. "
            "However, there are also growing concerns about job displacement and data privacy."
        ),
        (
            "Vietnam is a beautiful country located in Southeast Asia with a rich history "
            "and vibrant culture. The country stretches over one thousand kilometres from "
            "north to south. Vietnamese cuisine is renowned for its fresh ingredients, "
            "aromatic herbs, and bold flavours."
        ),
        (
            "Climate change is one of the most pressing challenges facing humanity today. "
            "Rising global temperatures are causing more frequent and severe weather events. "
            "Governments and scientists are working together to develop renewable energy "
            "solutions and reduce carbon emissions."
        ),
        (
            "Education is the foundation of a prosperous society. Access to quality education "
            "empowers individuals to reach their full potential. Modern technology has opened "
            "new avenues for learning, allowing students to access resources from anywhere "
            "in the world."
        ),
        (
            "The global economy has undergone significant changes over the past decade. "
            "International trade flows have increased dramatically due to advances in "
            "transportation and communication technology. Emerging markets in Asia and "
            "Africa are playing an increasingly important role in the world economy."
        ),
    ],
    "vi-en": [
        (
            "Trí tuệ nhân tạo đang thay đổi nhiều ngành công nghiệp trong những năm gần đây. "
            "Các công ty đang đầu tư mạnh mẽ vào nghiên cứu học máy và cơ sở hạ tầng. "
            "Những công nghệ này có thể giúp tự động hóa các tác vụ phức tạp. Tuy nhiên, "
            "cũng có những lo ngại ngày càng tăng về việc mất việc làm và quyền riêng tư."
        ),
        (
            "Việt Nam là một đất nước tươi đẹp nằm ở Đông Nam Á với lịch sử phong phú và "
            "văn hóa sôi động. Đất nước trải dài hơn một nghìn kilômét từ bắc vào nam. "
            "Ẩm thực Việt Nam nổi tiếng với nguyên liệu tươi và hương vị đậm đà."
        ),
        (
            "Biến đổi khí hậu là một trong những thách thức cấp bách nhất mà nhân loại "
            "đang phải đối mặt. Nhiệt độ toàn cầu tăng cao đang gây ra các hiện tượng "
            "thời tiết ngày càng khắc nghiệt. Các chính phủ đang hợp tác để phát triển "
            "các giải pháp năng lượng tái tạo và giảm lượng khí thải carbon."
        ),
        (
            "Giáo dục là nền tảng của một xã hội phồn thịnh. Tiếp cận giáo dục chất lượng "
            "trao quyền cho cá nhân phát huy hết tiềm năng của mình. Công nghệ hiện đại "
            "đã mở ra những con đường học tập mới, cho phép học sinh tiếp cận tài nguyên "
            "giáo dục từ bất kỳ đâu trên thế giới."
        ),
        (
            "Nền kinh tế toàn cầu đã trải qua những thay đổi đáng kể trong thập kỷ qua. "
            "Thương mại và đầu tư quốc tế đã tăng trưởng mạnh mẽ nhờ những tiến bộ trong "
            "công nghệ vận tải và truyền thông. Các thị trường mới nổi ở châu Á và châu Phi "
            "đang đóng vai trò ngày càng quan trọng."
        ),
    ],
}

_LONG: dict[str, list[str]] = {
    "en-vi": [
        (
            "Artificial intelligence and machine learning have become transformative forces "
            "in the modern world. Over the past decade, breakthroughs in deep learning have "
            "enabled computers to perform tasks once thought exclusively within the domain of "
            "human intelligence — recognising images, understanding speech, and translating "
            "languages. Large language models trained on vast amounts of text data can now "
            "generate coherent and contextually appropriate responses to a wide range of "
            "questions. These advances are driving innovation across industries, from "
            "healthcare and finance to transportation and entertainment. However, the rapid "
            "pace of AI development also raises important ethical questions about bias, "
            "transparency, and accountability."
        ),
        (
            "Vietnam's economic transformation over the past three decades is one of the "
            "most remarkable development stories in modern history. Since the introduction "
            "of the Doi Moi reforms in 1986, the country has shifted from a centrally planned "
            "economy to a market-oriented one, resulting in rapid GDP growth and a dramatic "
            "reduction in poverty. Millions of Vietnamese citizens have been lifted out of "
            "poverty, and the country has become an important hub for manufacturing and "
            "export in Southeast Asia. Foreign direct investment has poured in as "
            "multinational companies seek to diversify their supply chains. At the same time, "
            "Vietnam is investing heavily in education, technology, and infrastructure."
        ),
        (
            "The transition to renewable energy is one of the defining challenges of the "
            "twenty-first century. Fossil fuels, which have powered economic growth for over "
            "a century, are the primary driver of climate change. Solar and wind power have "
            "become dramatically cheaper in recent years, making them increasingly competitive "
            "with coal and natural gas. Battery storage technology is improving rapidly, "
            "helping to address the intermittency problem that has historically limited "
            "renewable sources. Electric vehicles are gaining market share, reducing "
            "dependence on petroleum in the transportation sector."
        ),
    ],
    "vi-en": [
        (
            "Trí tuệ nhân tạo và học máy đã trở thành những lực lượng chuyển đổi trong "
            "thế giới hiện đại. Trong thập kỷ qua, những đột phá trong học sâu đã cho phép "
            "máy tính thực hiện các nhiệm vụ từng được cho là thuộc về lĩnh vực trí tuệ con "
            "người — nhận dạng hình ảnh, hiểu giọng nói và dịch ngôn ngữ. Các mô hình ngôn "
            "ngữ lớn được đào tạo trên lượng dữ liệu văn bản khổng lồ hiện có thể tạo ra "
            "các phản hồi mạch lạc và phù hợp với ngữ cảnh. Những tiến bộ này đang thúc "
            "đẩy đổi mới trong các ngành công nghiệp, từ y tế và tài chính đến vận tải."
        ),
        (
            "Sự chuyển đổi kinh tế của Việt Nam trong ba thập kỷ qua là một trong những "
            "câu chuyện phát triển đáng chú ý nhất trong lịch sử hiện đại. Kể từ khi thực "
            "hiện cải cách Đổi Mới năm 1986, đất nước đã chuyển từ nền kinh tế kế hoạch tập "
            "trung sang nền kinh tế định hướng thị trường, dẫn đến tăng trưởng GDP nhanh "
            "chóng và giảm nghèo đói đáng kể. Hàng triệu người dân Việt Nam đã thoát khỏi "
            "đói nghèo và đất nước đã trở thành trung tâm quan trọng về sản xuất và xuất "
            "khẩu ở Đông Nam Á."
        ),
        (
            "Quá trình chuyển đổi sang năng lượng tái tạo là một trong những thách thức "
            "định nghĩa của thế kỷ hai mươi mốt. Nhiên liệu hóa thạch, đã thúc đẩy tăng "
            "trưởng kinh tế trong hơn một thế kỷ, là nguyên nhân chính gây ra biến đổi khí "
            "hậu. Năng lượng mặt trời và gió đã trở nên rẻ hơn đáng kể trong những năm gần "
            "đây. Công nghệ lưu trữ pin đang cải thiện nhanh chóng, giúp giải quyết vấn đề "
            "gián đoạn đã từng hạn chế độ tin cậy của các nguồn tái tạo."
        ),
    ],
}

_TEXT_POOLS = {"short": _SHORT, "medium": _MEDIUM, "long": _LONG}

# ---------------------------------------------------------------------------
# URL resolution — prefer IPv6 when IPv4 loopback doesn't reach Docker
# ---------------------------------------------------------------------------

def _resolve_api_url(api_url: str) -> str:
    """Return a URL that actually reaches the app.

    On some Linux hosts the Docker IPv4 userland proxy doesn't forward
    correctly while the IPv6 proxy works fine.  When the caller passes
    http://localhost:… we try both address families and return the one
    that responds with HTTP 200 on /health, preferring IPv4 first.
    """
    from urllib.parse import urlparse, urlunparse
    parsed = urlparse(api_url)
    host = parsed.hostname or "localhost"

    # Only attempt fallback for symbolic hostnames, not explicit IPs.
    if host not in ("localhost", "::1", "127.0.0.1"):
        return api_url

    port = parsed.port or 80

    for af, bind in [(socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")]:
        candidate_host = f"[::1]" if af == socket.AF_INET6 else "127.0.0.1"
        candidate = urlunparse(parsed._replace(netloc=f"{candidate_host}:{port}"))
        try:
            with urllib.request.urlopen(f"{candidate}/health", timeout=3) as resp:
                if resp.status == 200:
                    return candidate
        except Exception:
            pass

    return api_url  # fall through — caller will get a clean error


# ---------------------------------------------------------------------------
# App HTTP helpers
# ---------------------------------------------------------------------------

def _check_app_ready(api_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{api_url}/health", timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def _translate_one(
    api_url: str,
    api_key: str,
    direction: str,
    text: str,
    model: str | None,
    poll_interval: float,
    timeout: float,
) -> str:
    headers = {"Content-Type": "application/json", "X-API-Key": api_key}
    body: dict = {"text": text, "direction": direction}
    if model:
        body["model"] = model

    req = urllib.request.Request(
        f"{api_url}/translate",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    # Retry on 429 (Traefik rate limit) with exponential backoff.
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=min(timeout, 15.0)) as resp:
                job_id = json.loads(resp.read())["job_id"]
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < 4:
                time.sleep(0.5 * (2 ** attempt))
                continue
            raise RuntimeError(f"POST /translate → {exc.code}: {exc.read().decode()}") from exc

    poll_req = urllib.request.Request(f"{api_url}/jobs/{job_id}", headers={"X-API-Key": api_key})
    deadline = time.monotonic() + timeout
    rate_limit_backoff = poll_interval
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(poll_req, timeout=10.0) as resp:
                data = json.loads(resp.read())
            rate_limit_backoff = poll_interval  # reset on success
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                time.sleep(rate_limit_backoff)
                rate_limit_backoff = min(rate_limit_backoff * 2, 4.0)
                continue
            raise RuntimeError(f"GET /jobs/{job_id} → {exc.code}") from exc
        status = data.get("status")
        if status == "done":
            return str(data.get("translation", ""))
        if status == "failed":
            raise RuntimeError(f"job failed: {data.get('error')}")
        time.sleep(poll_interval)
    raise TimeoutError(f"job {job_id} not done after {timeout}s")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _worker(fn: Callable[[str], str], texts: list[str], idx: int) -> tuple[float, str | None]:
    text = texts[idx % len(texts)]
    t0 = time.perf_counter()
    try:
        fn(text)
        return time.perf_counter() - t0, None
    except Exception as exc:
        return time.perf_counter() - t0, str(exc)


def run_phase(
    api_url: str,
    api_key: str,
    direction: str,
    model: str | None,
    texts: list[str],
    n_requests: int,
    concurrency: int,
    poll_interval: float,
    job_timeout: float,
) -> dict:
    fn = lambda text: _translate_one(  # noqa: E731
        api_url, api_key, direction, text, model, poll_interval, job_timeout
    )

    latencies: list[float] = []
    errors: list[str] = []

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_worker, fn, texts, i): i for i in range(n_requests)}
        for fut in as_completed(futures):
            latency, err = fut.result()
            latencies.append(latency)
            if err:
                errors.append(err)

    return {
        "direction": direction,
        "backend": "llm" if model else "nmt",
        "model": model or "triton-nmt",
        "n_requests": n_requests,
        "concurrency": concurrency,
        "total_s": time.perf_counter() - t_start,
        "latencies": latencies,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _pct(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _section(r: dict, width: int) -> list[str]:
    n = r["n_requests"]
    n_err = len(r["errors"])
    n_ok = n - n_err
    lats = r["latencies"]
    lines: list[str] = []

    backend_tag = r["backend"].upper()
    lines.append(f"  [{backend_tag}]  {r['direction']}  —  {r['model']}")
    lines.append(f"  Requests    {n}  (concurrency={r['concurrency']})")
    lines.append(f"  Succeeded   {n_ok}     Failed: {n_err}")
    lines.append(f"  Duration    {r['total_s']:.2f} s")
    if r["total_s"] > 0:
        lines.append(f"  Throughput  {n / r['total_s']:.2f} req/s")
    if lats:
        lines.append(
            f"  Latency     min {min(lats):.3f}  mean {statistics.mean(lats):.3f}  "
            f"p50 {_pct(lats,50):.3f}  p95 {_pct(lats,95):.3f}  max {max(lats):.3f}  (s)"
        )
    if r["errors"]:
        lines.append(f"  First error: {r['errors'][0][:120]}")
    return lines


def build_report(phases: list[dict], api_url: str, text_size: str) -> str:
    W = 72
    lines: list[str] = [
        "=" * W,
        "  APP STRESS TEST REPORT",
        f"  target   : {api_url}",
        f"  text_size: {text_size}",
        f"  run_at   : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * W,
    ]

    current_phase = ""
    for r in phases:
        phase = "Phase 1 — NMT (Triton)" if r["backend"] == "nmt" else "Phase 2 — LLM (vLLM)"
        if phase != current_phase:
            lines.append("")
            lines.append(phase)
            lines.append("-" * W)
            current_phase = phase
        lines.extend(_section(r, W))
        lines.append("")

    # Summary table
    lines.append("=" * W)
    lines.append("  SUMMARY")
    lines.append(f"  {'backend':<8} {'direction':<8} {'req':>5} {'ok':>5} {'fail':>5} "
                 f"{'rps':>7} {'p50(s)':>7} {'p95(s)':>7}")
    lines.append("  " + "-" * 65)
    for r in phases:
        n = r["n_requests"]
        n_err = len(r["errors"])
        n_ok = n - n_err
        rps = n / r["total_s"] if r["total_s"] > 0 else 0.0
        lats = r["latencies"]
        p50 = _pct(lats, 50) if lats else 0.0
        p95 = _pct(lats, 95) if lats else 0.0
        lines.append(
            f"  {r['backend']:<8} {r['direction']:<8} {n:>5} {n_ok:>5} {n_err:>5} "
            f"{rps:>7.2f} {p50:>7.3f} {p95:>7.3f}"
        )
    lines.append("=" * W)

    return "\n".join(lines)


def _write_json(phases: list[dict], path: Path, api_url: str, text_size: str) -> None:
    def _summarise(r: dict) -> dict:
        lats = r["latencies"]
        out = {k: v for k, v in r.items() if k != "latencies"}
        if lats:
            out["latency_stats"] = {
                "min_s":  round(min(lats), 4),
                "mean_s": round(statistics.mean(lats), 4),
                "p50_s":  round(_pct(lats, 50), 4),
                "p95_s":  round(_pct(lats, 95), 4),
                "p99_s":  round(_pct(lats, 99), 4),
                "max_s":  round(max(lats), 4),
            }
        return out

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "api_url": api_url,
                "text_size": text_size,
                "phases": [_summarise(r) for r in phases],
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )
        fh.write("\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- target ---
    parser.add_argument("--api-url", default="http://localhost:80", metavar="URL",
                        help="FastAPI base URL (default: http://localhost:80)")
    parser.add_argument("--api-key", default="changeme", metavar="KEY",
                        help="X-API-Key header value")
    parser.add_argument("--backend", choices=["nmt", "llm", "both"], default="both",
                        help="Which backend to test (default: both)")
    parser.add_argument("--direction", choices=["en-vi", "vi-en", "both"], default="both",
                        help="Translation direction(s) to test (default: both)")
    parser.add_argument("--text-size", choices=["short", "medium", "long"], default="short",
                        help="Input length preset (default: short)")

    # --- NMT tuning ---
    nmt = parser.add_argument_group("NMT / Triton  (fast ONNX seq2seq)")
    nmt.add_argument("--nmt-requests", type=int, default=100, metavar="N",
                     help="Total requests per direction for NMT (default: 100)")
    nmt.add_argument("--nmt-concurrency", type=int, default=8, metavar="N",
                     help="Concurrent workers for NMT (default: 8)")

    # --- LLM tuning ---
    llm = parser.add_argument_group("LLM / vLLM  (autoregressive, slower)")
    llm.add_argument("--llm-requests", type=int, default=20, metavar="N",
                     help="Total requests per direction for LLM (default: 20)")
    llm.add_argument("--llm-concurrency", type=int, default=4, metavar="N",
                     help="Concurrent workers for LLM (default: 4)")
    llm.add_argument("--llm-model", default="llm", metavar="KEY",
                     help="Model key for the LLM backend (default: llm)")

    # --- polling ---
    parser.add_argument("--poll-interval", type=float, default=0.5, metavar="S",
                        help="Seconds between job-status polls (default: 0.5)")
    parser.add_argument("--job-timeout", type=float, default=120.0, metavar="S",
                        help="Seconds before a single job is considered timed-out (default: 120)")

    # --- output ---
    parser.add_argument("--results-file", metavar="JSON",
                        help="Write full results to this JSON file")
    parser.add_argument("--output", metavar="TXT",
                        help="Write text report to this file in addition to stdout")

    args = parser.parse_args()
    api_url = _resolve_api_url(args.api_url.rstrip("/"))

    # --- validate ---
    for flag, val in [("--nmt-requests", args.nmt_requests),
                      ("--llm-requests", args.llm_requests),
                      ("--nmt-concurrency", args.nmt_concurrency),
                      ("--llm-concurrency", args.llm_concurrency)]:
        if val <= 0:
            print(f"{flag} must be > 0", file=sys.stderr)
            raise SystemExit(1)

    print(f"[app-stress] Checking {api_url}/health …", file=sys.stderr)
    if not _check_app_ready(api_url):
        print(f"FastAPI app at {api_url} is not reachable. Is it running?", file=sys.stderr)
        raise SystemExit(1)
    print("[app-stress] App is up.", file=sys.stderr)

    pool = _TEXT_POOLS[args.text_size]
    directions = ["en-vi", "vi-en"] if args.direction == "both" else [args.direction]
    phases: list[dict] = []

    # ── Phase 1: NMT ─────────────────────────────────────────────────────────
    if args.backend in ("nmt", "both"):
        print(
            f"\n[Phase 1 — NMT]  {args.nmt_requests} req × {len(directions)} direction(s)"
            f"  concurrency={args.nmt_concurrency}  text={args.text_size}",
            file=sys.stderr,
        )
        for direction in directions:
            texts = pool[direction]
            print(f"  {direction} …", file=sys.stderr, flush=True)
            result = run_phase(
                api_url=api_url,
                api_key=args.api_key,
                direction=direction,
                model=None,
                texts=texts,
                n_requests=args.nmt_requests,
                concurrency=args.nmt_concurrency,
                poll_interval=args.poll_interval,
                job_timeout=args.job_timeout,
            )
            n_err = len(result["errors"])
            print(
                f"    done — {args.nmt_requests - n_err}/{args.nmt_requests} ok"
                f"  {result['total_s']:.1f}s"
                f"  {args.nmt_requests / result['total_s']:.1f} req/s",
                file=sys.stderr,
            )
            phases.append(result)

    # ── Phase 2: LLM ─────────────────────────────────────────────────────────
    if args.backend in ("llm", "both"):
        print(
            f"\n[Phase 2 — LLM]  {args.llm_requests} req × {len(directions)} direction(s)"
            f"  concurrency={args.llm_concurrency}  text={args.text_size}",
            file=sys.stderr,
        )
        for direction in directions:
            texts = pool[direction]
            print(f"  {direction} …", file=sys.stderr, flush=True)
            result = run_phase(
                api_url=api_url,
                api_key=args.api_key,
                direction=direction,
                model=args.llm_model,
                texts=texts,
                n_requests=args.llm_requests,
                concurrency=args.llm_concurrency,
                poll_interval=args.poll_interval,
                job_timeout=args.job_timeout,
            )
            n_err = len(result["errors"])
            print(
                f"    done — {args.llm_requests - n_err}/{args.llm_requests} ok"
                f"  {result['total_s']:.1f}s"
                f"  {args.llm_requests / result['total_s']:.1f} req/s",
                file=sys.stderr,
            )
            phases.append(result)

    if not phases:
        print("Nothing to run.", file=sys.stderr)
        raise SystemExit(1)

    report = build_report(phases, api_url, args.text_size)
    print(f"\n{report}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report + "\n", encoding="utf-8")
        print(f"\nReport saved → {args.output}", file=sys.stderr)

    if args.results_file:
        _write_json(phases, Path(args.results_file), api_url, args.text_size)
        print(f"Results saved → {args.results_file}", file=sys.stderr)

    # Exit non-zero if any phase had errors
    total_errors = sum(len(r["errors"]) for r in phases)
    if total_errors:
        print(f"\n[app-stress] {total_errors} error(s) total — see report above.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
