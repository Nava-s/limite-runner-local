"""Compare prompt formats on a running llama-server without changing its setup."""

import argparse
import json
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--question", default="Give me an explanation of Taylor Series.")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--out", type=Path, default=Path("experiments/template-comparison-2026-10-12.json"))
    args = parser.parse_args()

    def request(path, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(args.url.rstrip("/") + path, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as response:
            return json.load(response)

    props = request("/props")
    rendered = request("/apply-template", {
        "messages": [{"role": "user", "content": args.question}],
        "add_generation_prompt": True,
    })["prompt"]
    # Preserve the exact rendered user/assistant delimiters and initial BOS;
    # remove only the system turn. Do not edit the original template artifact.
    user_at = rendered.index("<|im_start|>user")
    system_at = rendered.index("<|im_start|>system")
    without_system = rendered[:system_at] + rendered[user_at:]
    modes = {
        "original_template": rendered,
        "chat_without_system": without_system,
        "raw_text": args.question,
    }
    settings = {
        "temperature": 0.0, "top_p": 0.95, "seed": 42,
        "n_predict": args.max_tokens, "cache_prompt": False,
        "stream": False, "stop": ["<|endoftext|>", "<|im_end|>"],
    }
    report = {
        "question": args.question, "settings": settings,
        "build": props.get("build_info"), "model": props.get("model_path"),
        "method": "One single-turn greedy generation per format; no prompt-cache reuse. Experimental formats are not the supported canonical chat contract.",
        "results": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for name, prompt in modes.items():
        print("Running " + name, flush=True)
        start = time.monotonic()
        response = request("/completion", {**settings, "prompt": prompt})
        result = {
            "mode": name, "prompt": prompt, "text": response.get("content", ""),
            "tokens": response.get("tokens_predicted"),
            "prompt_tokens": response.get("tokens_evaluated"),
            "stop_type": response.get("stop_type"),
            "finish_reason": "length" if response.get("stop_type") == "limit" or response.get("truncated") else "stop",
            "seconds": round(time.monotonic() - start, 3),
            "timings": response.get("timings"),
        }
        report["results"].append(result)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{name}: {result['tokens']} tokens, {result['finish_reason']}, {result['seconds']} s", flush=True)
    readable = []
    for result in report["results"]:
        readable.append(f"=== {result['mode']} ===\n{result['tokens']} tokens | {result['finish_reason']} | {result['seconds']} s\n\n{result['text']}\n")
    args.out.with_suffix(".txt").write_text("\n".join(readable), encoding="utf-8")


if __name__ == "__main__":
    main()
