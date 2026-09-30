"""Run a small, reproducible paired benchmark against the running local server."""
import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jev_local.examples import EXAMPLES


def call(lane, body):
    request = Request(f"http://127.0.0.1:8765/api/{lane}",
                      data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=180) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmark-results.json")
    args = parser.parse_args()
    first = {k: EXAMPLES[0][k] for k in ("state", "question", "choices")}
    print("Warming both paths; warmup excluded from reported timings.", flush=True)
    call("decide", first)
    call("generate", {**first, "max_tokens": 80})
    rows = []
    for i, case in enumerate(EXAMPLES):
        body = {k: case[k] for k in ("state", "question", "choices")}
        row = {"id": case["id"], "expected": case["expected"]}
        for lane in (("decide", "generate") if i % 2 == 0 else ("generate", "decide")):
            result = call(lane, body if lane == "decide" else {**body, "max_tokens": 80})
            if lane == "decide":
                assert abs(sum(result["probabilities"].values()) - 1) < 1e-5
                assert result["output_tokens"] == 0
                assert result["choice"] in result["probabilities"]
            row[lane] = {**result, "correct": result["choice"] == case["expected"]}
        rows.append(row)
        print(f'{i+1:2}/{len(EXAMPLES)} {case["title"]}: '
              f'score={row["decide"]["choice"]} ({row["decide"]["latency_ms"]:.0f} ms), '
              f'generate={row["generate"]["choice"]} ({row["generate"]["latency_ms"]:.0f} ms)', flush=True)
    summary = {lane: {
        "correct": sum(r[lane]["correct"] for r in rows), "cases": len(rows),
        "mean_ms": round(statistics.mean(r[lane]["latency_ms"] for r in rows), 2),
        "median_ms": round(statistics.median(r[lane]["latency_ms"] for r in rows), 2),
        "output_tokens": sum(r[lane]["output_tokens"] for r in rows),
    } for lane in ("decide", "generate")}
    summary["mean_latency_ratio"] = round(summary["generate"]["mean_ms"] / summary["decide"]["mean_ms"], 2)
    report = {"timestamp": datetime.now(timezone.utc).isoformat(),
              "method": "12 synthetic support tickets. Sequential pairs, alternating order, both paths warmed. Same model; task-specific prompts. No prompt caching. Generation capped at 80 tokens. Timings exclude HTTP overhead. This is a smoke benchmark, not a general accuracy evaluation.",
              "summary": summary, "results": rows}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
