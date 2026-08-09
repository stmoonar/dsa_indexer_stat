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


def fit_to_budget(tok, text, budget):
    """把文本按 token 截断到 budget，返回 (text, n_tokens)。"""
    ids = tok.encode(text)
    if len(ids) <= budget:
        return text, len(ids)
    text = tok.decode(ids[:budget], skip_special_tokens=True)
    return text, len(tok.encode(text))


def build_sequences(cands, tok, budget, n_samples, allow_concat, fill_ratio=0.9):
    """从候选里挑出 n_samples 条互不重复的序列。

    单条样本够长 → 直接用（必要时截断到 budget）。
    都不够长且允许拼接 → 顺序拼多条直到填满 budget。
    拼接出来的不是自然长文，语义上更接近 multi-doc，manifest 里会标出来，
    分析时不能当成"真实 128K agentic trace"解读。
    """
    scored = []
    for char_len, path, row_idx, text in cands:
        n = len(tok.encode(text))
        scored.append({"n": n, "path": path, "row": row_idx, "text": text})
        logger.info(f"  {os.path.basename(path)}[{row_idx}]: "
                    f"{char_len} chars → {n} tokens")
    scored.sort(key=lambda e: -e["n"])

    used, seqs = set(), []
    for i in range(n_samples):
        pool = [e for j, e in enumerate(scored) if j not in used]
        if not pool:
            logger.warning(f"Ran out of candidates after {len(seqs)} sequences")
            break

        # 先看有没有单条就够长的（取最接近 budget 的那条）
        single = min(pool, key=lambda e: abs(e["n"] - budget))
        idx = scored.index(single)
        if single["n"] >= int(budget * fill_ratio):
            text, n = fit_to_budget(tok, single["text"], budget)
            used.add(idx)
            seqs.append({"text": text, "n_tokens": n,
                         "source_file": os.path.basename(single["path"]),
                         "row_index": single["row"],
                         "truncated": n < single["n"], "concat_parts": 1})
            continue

        if not allow_concat:
            text, n = fit_to_budget(tok, single["text"], budget)
            used.add(idx)
            seqs.append({"text": text, "n_tokens": n,
                         "source_file": os.path.basename(single["path"]),
                         "row_index": single["row"],
                         "truncated": n < single["n"], "concat_parts": 1,
                         "underfilled": True})
            logger.warning(f"seq{i:02d} only {n}/{budget} tokens "
                           f"(pass --concat to fill)")
            continue

        # 拼接模式：按长度降序吃候选，直到填满 budget
        parts, total, srcs = [], 0, []
        for j, e in enumerate(scored):
            if j in used:
                continue
            parts.append(e["text"])
            srcs.append(f"{os.path.basename(e['path'])}[{e['row']}]")
            used.add(j)
            total += e["n"]
            if total >= budget:
                break
        text, n = fit_to_budget(tok, "\n\n".join(parts), budget)
        seqs.append({"text": text, "n_tokens": n,
                     "source_file": ",".join(srcs), "row_index": -1,
                     "truncated": True, "concat_parts": len(parts)})
        logger.info(f"seq{i:02d}: concatenated {len(parts)} traces → {n} tokens")
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--tokenizer", required=True,
                    help="模型目录（用其 tokenizer 精确计数）")
    ap.add_argument("--context-length", type=int, default=32768)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--margin", type=int, default=64,
                    help="预留给 chat template / BOS 等的 token 余量")
    ap.add_argument("--column", default=None, help="文本列名；缺省自动探测")
    ap.add_argument("--n-samples", type=int, default=1,
                    help="生成几条互不重复的序列（序列间方差需要 >1）")
    ap.add_argument("--concat", action="store_true",
                    help="单条样本不够长时拼接多条填满 budget（128K 必需）")
    ap.add_argument("--out-prefix", required=True,
                    help="输出前缀，产出 <prefix>_seq00.txt 和 <prefix>_manifest.json")
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
                f"token budget = {budget}, n_samples = {args.n_samples}")

    # 拼接模式要多备候选；非拼接也多留几条以便挑出 n_samples 条不重复的
    keep_top = max(8, args.n_samples * (12 if args.concat else 4))
    candidates = scan_candidates(
        parquet_files, args.column,
        target_chars=int(budget * CHARS_PER_TOKEN_EST), keep_top=keep_top)
    logger.info(f"Coarse-filtered {len(candidates)} candidates, tokenizing ...")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    seqs = build_sequences(candidates, tok, budget, args.n_samples, args.concat)
    if not seqs:
        raise RuntimeError("No sequence could be built; dataset too short?")

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    entries = []
    for i, s in enumerate(seqs):
        path = f"{args.out_prefix}_seq{i:02d}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(s["text"])
        entries.append({k: v for k, v in s.items() if k != "text"} |
                       {"path": path, "name": f"seq{i:02d}"})
        logger.info(f"Wrote {path} ({s['n_tokens']} tokens)")

    manifest = {
        "token_budget": budget,
        "context_length": args.context_length,
        "max_new_tokens": args.max_new_tokens,
        "tokenizer": args.tokenizer,
        "concat_enabled": args.concat,
        "sequences": entries,
    }
    mpath = f"{args.out_prefix}_manifest.json"
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Wrote manifest {mpath}")
    logger.info(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
