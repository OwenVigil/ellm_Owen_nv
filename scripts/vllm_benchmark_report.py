#!/usr/bin/env python3
"""Create a static HTML report from vLLM benchmark_serving result JSON files."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    value = as_float(value, float("nan"))
    if math.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}{suffix}"


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * p / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def clean_numbers(values: Iterable[Any]) -> list[float]:
    cleaned: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and number > 0:
            cleaned.append(number)
    return cleaned


GPU_KV_CACHE = "gpu_kv_cache_usage_percent"
CPU_KV_CACHE = "cpu_kv_cache_usage_percent"


def tpot_ms_from_itls(result: dict[str, Any]) -> list[float]:
    tpots: list[float] = []
    for intervals in result.get("itls", []):
        if not isinstance(intervals, list):
            continue
        values = clean_numbers(intervals)
        if values:
            tpots.append(mean(values) * 1000.0)
    return tpots


def server_metric_samples(result: dict[str, Any]) -> list[dict[str, float]]:
    server_metrics = result.get("server_metrics", {})
    raw_samples = (server_metrics.get("samples", [])
                   if isinstance(server_metrics, dict) else [])
    samples: list[dict[str, float]] = []
    for raw_sample in raw_samples:
        if not isinstance(raw_sample, dict):
            continue
        time_s = as_float(raw_sample.get("time_s"), float("nan"))
        sample: dict[str, float] = {"time_s": time_s}
        for metric_name in (GPU_KV_CACHE, CPU_KV_CACHE):
            value = as_float(raw_sample.get(metric_name), float("nan"))
            if math.isfinite(value):
                sample[metric_name] = value
        if math.isfinite(time_s) and time_s >= 0 and len(sample) > 1:
            samples.append(sample)
    return samples


def server_metric_stat(result: dict[str, Any], metric_name: str,
                       statistic: str) -> float:
    server_metrics = result.get("server_metrics", {})
    if isinstance(server_metrics, dict):
        summary = server_metrics.get("summary", {})
        if isinstance(summary, dict):
            metric_summary = summary.get(metric_name, {})
            if isinstance(metric_summary, dict):
                stored = as_float(metric_summary.get(statistic), float("nan"))
                if math.isfinite(stored):
                    return stored

    values = [
        sample[metric_name] for sample in server_metric_samples(result)
        if metric_name in sample
    ]
    if not values:
        return float("nan")
    if statistic == "mean":
        return mean(values)
    if statistic == "p95":
        return percentile(values, 95)
    return max(values)


def load_result(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        result = json.load(handle)
    result["_path"] = str(path)
    return result


def make_histogram(values: list[float], bins: int = 18) -> list[tuple[float, float, int]]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if lo == hi:
        return [(lo, hi, len(values))]

    bins = max(1, min(bins, len(values)))
    width = (hi - lo) / bins
    counts = [0] * bins
    for value in values:
        index = min(int((value - lo) / width), bins - 1)
        counts[index] += 1
    return [(lo + i * width, lo + (i + 1) * width, counts[i]) for i in range(bins)]


def histogram_svg(values: list[float], title: str, color: str) -> str:
    if not values:
        return (
            f"<section class='panel'><h2>{html.escape(title)}</h2>"
            "<p class='empty'>No successful samples.</p></section>"
        )

    buckets = make_histogram(values)
    max_count = max(count for _, _, count in buckets) or 1
    width = 780
    height = 280
    left = 54
    right = 18
    top = 34
    bottom = 46
    plot_w = width - left - right
    plot_h = height - top - bottom
    bar_gap = 3
    bar_w = max(1, plot_w / len(buckets) - bar_gap)

    parts = [
        f"<section class='panel'><h2>{html.escape(title)}</h2>",
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)} histogram'>",
        f"<line x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' y2='{top + plot_h}' class='axis'/>",
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}' class='axis'/>",
    ]

    for i, (start, end, count) in enumerate(buckets):
        x = left + i * (plot_w / len(buckets)) + bar_gap / 2
        bar_h = count / max_count * plot_h
        y = top + plot_h - bar_h
        label = f"{start:.1f}-{end:.1f} ms: {count}"
        parts.append(
            f"<rect x='{x:.2f}' y='{y:.2f}' width='{bar_w:.2f}' "
            f"height='{bar_h:.2f}' fill='{color}'><title>{html.escape(label)}</title></rect>"
        )

    x_labels = [
        (left, min(values)),
        (left + plot_w / 2, percentile(values, 50)),
        (left + plot_w, max(values)),
    ]
    for x, value in x_labels:
        parts.append(
            f"<text x='{x:.2f}' y='{height - 18}' text-anchor='middle' class='tick'>{value:.1f}</text>"
        )
    parts.append(f"<text x='14' y='{top + 10}' class='tick'>{max_count}</text>")
    parts.append(
        f"<text x='{left + plot_w / 2}' y='{height - 4}' text-anchor='middle' class='tick'>milliseconds</text>"
    )
    parts.extend(["</svg>", "</section>"])
    return "\n".join(parts)


def kv_cache_svg(result: dict[str, Any]) -> str:
    samples = server_metric_samples(result)
    if not samples:
        return (
            "<section class='panel'><h2>KV Cache Usage</h2>"
            "<p class='empty'>Not collected for this run.</p></section>"
        )

    width = 780
    height = 300
    left = 58
    right = 22
    top = 34
    bottom = 48
    plot_w = width - left - right
    plot_h = height - top - bottom
    duration = as_float(result.get("duration"), 0.0)
    max_time = max([duration, 1.0, *[sample["time_s"] for sample in samples]])

    parts = [
        "<section class='panel'><div class='panel-title'>",
        "<h2>KV Cache Usage</h2>",
        "<div class='legend'><span class='gpu-key'>GPU</span>"
        "<span class='cpu-key'>CPU</span></div></div>",
        f"<svg viewBox='0 0 {width} {height}' role='img' "
        "aria-label='GPU and CPU KV cache usage over benchmark time'>",
    ]

    for percent in (0, 25, 50, 75, 100):
        y = top + plot_h - percent / 100.0 * plot_h
        parts.append(
            f"<line x1='{left}' y1='{y:.2f}' x2='{left + plot_w}' "
            f"y2='{y:.2f}' class='grid-line'/>"
        )
        parts.append(
            f"<text x='{left - 8}' y='{y + 4:.2f}' text-anchor='end' "
            f"class='tick'>{percent}%</text>"
        )

    parts.append(
        f"<line x1='{left}' y1='{top}' x2='{left}' "
        f"y2='{top + plot_h}' class='axis'/>"
    )
    parts.append(
        f"<line x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' "
        f"y2='{top + plot_h}' class='axis'/>"
    )

    for metric_name, css_class in ((GPU_KV_CACHE, "gpu-line"),
                                   (CPU_KV_CACHE, "cpu-line")):
        points = []
        for sample in samples:
            if metric_name not in sample:
                continue
            x = left + sample["time_s"] / max_time * plot_w
            plotted_value = min(100.0, max(0.0, sample[metric_name]))
            y = top + plot_h - plotted_value / 100.0 * plot_h
            points.append(f"{x:.2f},{y:.2f}")
        if points:
            parts.append(
                f"<polyline points='{' '.join(points)}' class='{css_class}'/>"
            )

    parts.append(
        f"<text x='{left}' y='{height - 18}' text-anchor='middle' "
        "class='tick'>0</text>"
    )
    parts.append(
        f"<text x='{left + plot_w}' y='{height - 18}' text-anchor='middle' "
        f"class='tick'>{max_time:.1f}</text>"
    )
    parts.append(
        f"<text x='{left + plot_w / 2}' y='{height - 4}' "
        "text-anchor='middle' class='tick'>benchmark time (seconds)</text>"
    )
    parts.extend(["</svg>", "</section>"])
    return "\n".join(parts)


def metric_cards(result: dict[str, Any]) -> str:
    cards = [
        ("Requests", result.get("completed"), "", 0),
        ("Duration", result.get("duration"), " s", 2),
        ("Req Throughput", result.get("request_throughput"), " req/s", 2),
        ("Input Throughput", result.get("input_throughput"), " tok/s", 1),
        ("Output Throughput", result.get("output_throughput"), " tok/s", 1),
        ("Mean TTFT", result.get("mean_ttft_ms"), " ms", 1),
        ("P99 TTFT", result.get("p99_ttft_ms"), " ms", 1),
        ("Mean TPOT", result.get("mean_tpot_ms"), " ms", 1),
        ("P99 TPOT", result.get("p99_tpot_ms"), " ms", 1),
    ]
    if isinstance(result.get("server_metrics"), dict):
        cards.extend([
            ("GPU KV Mean", server_metric_stat(
                result, GPU_KV_CACHE, "mean"), "%", 1),
            ("GPU KV P95", server_metric_stat(
                result, GPU_KV_CACHE, "p95"), "%", 1),
            ("GPU KV Max", server_metric_stat(
                result, GPU_KV_CACHE, "max"), "%", 1),
            ("CPU KV Max", server_metric_stat(
                result, CPU_KV_CACHE, "max"), "%", 1),
        ])
    rendered = []
    for label, value, suffix, digits in cards:
        rendered.append(
            "<div class='metric'>"
            f"<span>{html.escape(label)}</span>"
            f"<strong>{html.escape(fmt(value, digits, suffix))}</strong>"
            "</div>"
        )
    return "<div class='metrics'>" + "\n".join(rendered) + "</div>"


def length_svg(inputs: list[float], outputs: list[float]) -> str:
    pairs = [(x, y) for x, y in zip(inputs, outputs) if x > 0 and y > 0]
    if not pairs:
        return (
            "<section class='panel'><h2>Input vs Output Length</h2>"
            "<p class='empty'>No length samples.</p></section>"
        )

    width = 780
    height = 320
    left = 58
    right = 22
    top = 24
    bottom = 48
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_x = max(x for x, _ in pairs) or 1
    max_y = max(y for _, y in pairs) or 1

    parts = [
        "<section class='panel'><h2>Input vs Output Length</h2>",
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Input and output token length scatter plot'>",
        f"<line x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' y2='{top + plot_h}' class='axis'/>",
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}' class='axis'/>",
    ]
    for x_val, y_val in pairs[:2000]:
        x = left + x_val / max_x * plot_w
        y = top + plot_h - y_val / max_y * plot_h
        parts.append(
            f"<circle cx='{x:.2f}' cy='{y:.2f}' r='3.2' class='dot'>"
            f"<title>input {x_val:.0f}, output {y_val:.0f}</title></circle>"
        )
    parts.append(f"<text x='{left}' y='{height - 18}' text-anchor='middle' class='tick'>0</text>")
    parts.append(
        f"<text x='{left + plot_w}' y='{height - 18}' text-anchor='middle' class='tick'>{max_x:.0f}</text>"
    )
    parts.append(f"<text x='18' y='{top + 8}' class='tick'>{max_y:.0f}</text>")
    parts.append(
        f"<text x='{left + plot_w / 2}' y='{height - 4}' text-anchor='middle' class='tick'>input tokens</text>"
    )
    parts.extend(["</svg>", "</section>"])
    return "\n".join(parts)


def samples_table(result: dict[str, Any], limit: int = 6) -> str:
    texts = result.get("generated_texts", [])
    errors = result.get("errors", [])
    rows = []
    for idx, text in enumerate(texts[:limit]):
        error = errors[idx] if idx < len(errors) else ""
        preview = error or text
        preview = str(preview).replace("\n", " ")
        if len(preview) > 480:
            preview = preview[:480] + "..."
        status = "error" if error else "ok"
        rows.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td><span class='status {status}'>{status}</span></td>"
            f"<td>{html.escape(preview)}</td>"
            "</tr>"
        )
    if not rows:
        return ""
    return (
        "<section class='panel wide'><h2>Generated Text Preview</h2>"
        "<table><thead><tr><th>#</th><th>Status</th><th>Preview</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></section>"
    )


def run_section(result: dict[str, Any]) -> str:
    title = Path(result["_path"]).name
    ttfts_ms = [value * 1000.0 for value in clean_numbers(result.get("ttfts", []))]
    tpots_ms = tpot_ms_from_itls(result)
    input_lens = clean_numbers(result.get("input_lens", []))
    output_lens = clean_numbers(result.get("output_lens", []))
    errors = [err for err in result.get("errors", []) if err]
    meta = [
        f"model: {result.get('model_id', 'n/a')}",
        f"backend: {result.get('backend', 'n/a')}",
        f"request_rate: {result.get('request_rate', 'n/a')}",
        f"errors: {len(errors)}",
    ]
    return "\n".join(
        [
            "<article class='run'>",
            f"<h1>{html.escape(title)}</h1>",
            f"<p class='meta'>{html.escape(' | '.join(meta))}</p>",
            metric_cards(result),
            "<div class='grid'>",
            histogram_svg(ttfts_ms, "TTFT Distribution", "#2563eb"),
            histogram_svg(tpots_ms, "TPOT Distribution", "#059669"),
            kv_cache_svg(result),
            length_svg(input_lens, output_lens),
            samples_table(result),
            "</div>",
            "</article>",
        ]
    )


def comparison_table(results: list[dict[str, Any]]) -> str:
    if len(results) < 2:
        return ""
    rows = []
    for result in results:
        rows.append(
            "<tr>"
            f"<td>{html.escape(Path(result['_path']).name)}</td>"
            f"<td>{html.escape(str(result.get('request_rate', 'n/a')))}</td>"
            f"<td>{html.escape(fmt(result.get('request_throughput'), 2))}</td>"
            f"<td>{html.escape(fmt(result.get('output_throughput'), 1))}</td>"
            f"<td>{html.escape(fmt(result.get('mean_ttft_ms'), 1))}</td>"
            f"<td>{html.escape(fmt(result.get('p99_ttft_ms'), 1))}</td>"
            f"<td>{html.escape(fmt(result.get('mean_tpot_ms'), 1))}</td>"
            f"<td>{html.escape(fmt(result.get('p99_tpot_ms'), 1))}</td>"
            f"<td>{html.escape(fmt(server_metric_stat(result, GPU_KV_CACHE, 'mean'), 1))}</td>"
            f"<td>{html.escape(fmt(server_metric_stat(result, GPU_KV_CACHE, 'p95'), 1))}</td>"
            f"<td>{html.escape(fmt(server_metric_stat(result, GPU_KV_CACHE, 'max'), 1))}</td>"
            f"<td>{html.escape(fmt(server_metric_stat(result, CPU_KV_CACHE, 'max'), 1))}</td>"
            "</tr>"
        )
    return (
        "<section class='panel wide'><h2>Run Comparison</h2>"
        "<div class='table-scroll'><table><thead><tr>"
        "<th>file</th><th>request rate</th>"
        "<th>req/s</th><th>out tok/s</th><th>mean TTFT ms</th>"
        "<th>p99 TTFT ms</th><th>mean TPOT ms</th><th>p99 TPOT ms</th>"
        "<th>GPU KV mean %</th><th>GPU KV p95 %</th>"
        "<th>GPU KV max %</th><th>CPU KV max %</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>"
    )


def build_html(results: list[dict[str, Any]]) -> str:
    body = "\n".join([comparison_table(results), *[run_section(r) for r in results]])
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>vLLM Benchmark Report</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f8fafc;
  --panel: #ffffff;
  --text: #172033;
  --muted: #64748b;
  --border: #dbe3ef;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.45;
}}
main {{
  width: min(1180px, calc(100vw - 32px));
  margin: 28px auto 56px;
}}
h1 {{
  margin: 0;
  font-size: 24px;
  font-weight: 750;
}}
h2 {{
  margin: 0 0 14px;
  font-size: 16px;
  font-weight: 720;
}}
.run {{ margin-top: 26px; }}
.meta {{
  margin: 6px 0 18px;
  color: var(--muted);
  font-size: 13px;
}}
.metrics {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(148px, 1fr));
  gap: 10px;
  margin-bottom: 14px;
}}
.metric {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px;
}}
.metric span {{
  display: block;
  color: var(--muted);
  font-size: 12px;
}}
.metric strong {{
  display: block;
  margin-top: 5px;
  font-size: 18px;
}}
.grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
  gap: 14px;
}}
.panel {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px;
  overflow: hidden;
}}
.wide {{ grid-column: 1 / -1; }}
.panel-title {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}}
.legend {{
  display: flex;
  gap: 14px;
  color: var(--muted);
  font-size: 12px;
}}
.legend span::before {{
  content: "";
  display: inline-block;
  width: 18px;
  height: 3px;
  margin: 0 6px 3px 0;
}}
.gpu-key::before {{ background: #d97706; }}
.cpu-key::before {{ background: #7c3aed; }}
svg {{ width: 100%; height: auto; display: block; }}
.axis {{ stroke: #94a3b8; stroke-width: 1; }}
.grid-line {{ stroke: #e2e8f0; stroke-width: 1; }}
.gpu-line, .cpu-line {{ fill: none; stroke-width: 2.5; }}
.gpu-line {{ stroke: #d97706; }}
.cpu-line {{ stroke: #7c3aed; }}
.tick {{ fill: #64748b; font-size: 11px; }}
.dot {{ fill: #dc2626; opacity: 0.58; }}
.empty {{ color: var(--muted); }}
.table-scroll {{ overflow-x: auto; }}
table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}}
th {{ white-space: nowrap; }}
th, td {{
  border-bottom: 1px solid var(--border);
  padding: 8px 10px;
  text-align: left;
  vertical-align: top;
}}
th {{ color: var(--muted); font-weight: 650; }}
.status {{
  display: inline-block;
  border-radius: 999px;
  padding: 2px 8px;
  font-size: 12px;
}}
.status.ok {{ background: #dcfce7; color: #166534; }}
.status.error {{ background: #fee2e2; color: #991b1b; }}
@media (max-width: 760px) {{
  main {{ width: min(100vw - 20px, 1180px); margin-top: 18px; }}
  .grid {{ grid-template-columns: 1fr; }}
  .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
}}
</style>
</head>
<body>
<main>
{body}
</main>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a static HTML report from vLLM benchmark result JSON files."
    )
    parser.add_argument("results", nargs="+", type=Path, help="Result JSON files.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("vllm_benchmark_report.html"),
        help="Output HTML path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = [load_result(path) for path in args.results]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_html(results), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
