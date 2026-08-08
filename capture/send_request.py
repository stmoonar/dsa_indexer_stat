"""
send_request.py — 向 SGLang server 发送单条 capture 请求（bs=1，约束 7）。

用法：
    python -m capture.send_request \
        --prompt-file prompts/agentic_32k.txt \
        --max-new-tokens 256 --port 30000
"""

import json
import argparse
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read()

    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": args.max_new_tokens,
            # 贪心解码，保证 run 可复现（约束 13 的配套要求）
            "temperature": 0.0,
        },
    }

    url = f"http://{args.host}:{args.port}/generate"
    print(f"POST {url}  (prompt {len(prompt)} chars, "
          f"max_new_tokens={args.max_new_tokens})")

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    meta = result.get("meta_info", {})
    print(f"Done. completion_tokens={meta.get('completion_tokens')}, "
          f"prompt_tokens={meta.get('prompt_tokens')}")
    text = result.get("text", "")
    print(f"--- first 500 chars of output ---\n{text[:500]}")


if __name__ == "__main__":
    main()
