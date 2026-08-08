"""
prepare_prompt.py — 从 lmcache-agentic-traces parquet 数据集中挑选一条
接近目标上下文长度的样本，写成 capture 用的 prompt 文件。

用法（在有数据集和模型的机器上运行）：
    python -m capture.prepare_prompt \
        --dataset-dir /data/datasets/lmcache-agentic-traces/data \
        --tokenizer /data1/models/DeepSeek-V3.2 \
        --context-length 32768 --max-new-tokens 256 \
        --output prompts/agentic_32k.txt

策略：
1. 逐个 parquet 文件扫描，自动探测文本列（messages/conversations/prompt/text...）。
2. 先用 chars/token≈3.0 粗筛出接近目标 token 数的候选，只对候选做精确 tokenize。
3. 选 token 数最大但 ≤ budget 的样本；若最长样本仍超 budget，按 token 截断。
   budget = context_length - max_new_tokens - margin。

依赖: pip install pyarrow transformers
"""

import os
import json
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("prepare_prompt")

# 文本列探测优先级
TEXT_COLUMN_CANDIDATES = [
    "messages", "conversations", "conversation",
    "prompt", "text", "input", "content",
]

# 粗筛用的字符/token 比（对英文+代码混合 trace 偏保守）
CHARS_PER_TOKEN_EST = 3.0


def row_to_text(value) -> str:
    """把一行的文本列内容展平成纯文本 prompt。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    # messages 风格: list of {role, content}
    if isinstance(value, (list, tuple)):
        parts = []
        for item in value:
            if isinstance(item, dict):
                role = item.get("role", item.get("from", ""))
                content = item.get("content", item.get("value", ""))
                if isinstance(content, (list, tuple)):
                    # multimodal content list → 只取文本段
                    content = "\n".join(
                        c.get("text", "") if isinstance(c, dict) else str(c)
                        for c in content
                    )
                parts.append(f"{role}: {content}" if role else str(content))
            else:
                parts.append(str(item))
        return "\n\n".join(parts)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def detect_text_column(schema_names) -> str:
    for cand in TEXT_COLUMN_CANDIDATES:
        if cand in schema_names:
            return cand
    raise ValueError(
        f"No known text column found. Available columns: {list(schema_names)}. "
        f"Pass --column explicitly."
    )


def scan_candidates(parquet_files, column, target_chars, keep_top=8):
    """
    粗筛：按字符数找最接近 target_chars 的样本。
    返回 [(char_len, file, row_idx, text)]，按 |char_len - target| 升序取前 keep_top，
    并额外保留全局最长的一条（兜底截断用）。
    """
    import pyarrow.parquet as pq

    candidates = []   # (char_len, file, row_idx, text)
    longest = None

    for path in parquet_files:
        logger.info(f"Scanning {os.path.basename(path)} ...")
        pf = pq.ParquetFile(path)
        row_base = 0
        col = column
        if col is None:
            col = detect_text_column(pf.schema_arrow.names)
            logger.info(f"  Detected text column: '{col}'")
        for batch in pf.iter_batches(batch_size=256, columns=[col]):
            values = batch.column(0).to_pylist()
            for i, v in enumerate(values):
                text = row_to_text(v)
                n = len(text)
                entry = (n, path, row_base + i, text)
                if longest is None or n > longest[0]:
                    longest = entry
                candidates.append((abs(n - target_chars), entry))
            row_base += len(values)
        # 控制内存：每个文件扫完后只保留 top 候选
        candidates.sort(key=lambda x: x[0])
        candidates = candidates[:keep_top]

    result = [e for _, e in candidates]
    if longest is not None and all(e[2] != longest[2] or e[1] != longest[1]
                                   for e in result):
        result.append(longest)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--tokenizer", required=True,
                    help="模型目录（用其 tokenizer 精确计数）")
    ap.add_argument("--context-length", type=int, default=32768)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--margin", type=int, default=64,
                    help="预留给 chat template / BOS 等的 token 余量")
    ap.add_argument("--column", default=None,
                    help="文本列名；缺省自动探测")
    ap.add_argument("--output", required=True, help="prompt 输出路径 (.txt)")
    args = ap.parse_args()

    budget = args.context_length - args.max_new_tokens - args.margin
    assert budget > 0

    parquet_files = sorted(
        os.path.join(args.dataset_dir, f)
        for f in os.listdir(args.dataset_dir)
        if f.endswith(".parquet")
    )
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {args.dataset_dir}")
    logger.info(f"Found {len(parquet_files)} parquet files, "
                f"token budget = {budget}")

    candidates = scan_candidates(
        parquet_files, args.column, target_chars=int(budget * CHARS_PER_TOKEN_EST)
    )
    logger.info(f"Coarse-filtered {len(candidates)} candidates, tokenizing ...")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    best = None          # (n_tokens, file, row_idx, text) 满足 <= budget 中最大
    longest_tok = None   # 全部候选中 token 数最大者（兜底截断）
    for char_len, path, row_idx, text in candidates:
        ids = tok.encode(text)
        n = len(ids)
        logger.info(f"  {os.path.basename(path)}[{row_idx}]: "
                    f"{char_len} chars → {n} tokens")
        entry = (n, path, row_idx, text, ids)
        if longest_tok is None or n > longest_tok[0]:
            longest_tok = entry
        if n <= budget and (best is None or n > best[0]):
            best = entry

    truncated = False
    # 若最优样本填充率不足 90%，且存在超长样本，则截断超长样本以贴满 budget
    if (best is None or best[0] < int(budget * 0.9)) and longest_tok[0] > budget:
        n, path, row_idx, _, ids = longest_tok
        ids = ids[:budget]
        text = tok.decode(ids, skip_special_tokens=True)
        best = (len(tok.encode(text)), path, row_idx, text, ids)
        truncated = True
        logger.info(f"Truncated {os.path.basename(path)}[{row_idx}] "
                    f"from {n} to {best[0]} tokens")
    if best is None:
        raise RuntimeError("No sample fits the token budget; dataset too short?")

    n_tokens, path, row_idx, text, _ = best
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(text)

    meta = {
        "source_file": os.path.basename(path),
        "row_index": row_idx,
        "n_tokens": n_tokens,
        "token_budget": budget,
        "context_length": args.context_length,
        "max_new_tokens": args.max_new_tokens,
        "truncated": truncated,
        "tokenizer": args.tokenizer,
    }
    meta_path = args.output + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Wrote prompt ({n_tokens} tokens) to {args.output}")
    logger.info(f"Meta: {json.dumps(meta, indent=2)}")


if __name__ == "__main__":
    main()
