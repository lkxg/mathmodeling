"""Independent audio/transcript audit; never changes extraction inputs or features."""
from __future__ import annotations

import argparse
import csv
import fcntl
import html
import importlib.metadata
import inspect
import logging
import os
import re
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np
import soundfile as sf

from .common import atomic_json, digest, file_hash, load_config, read_json

QWEN_MODEL = {"id": "Qwen/Qwen3-ASR-1.7B-hf",
              "revision": "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"}
STATUS = {
    "large_difference": "文字差异较大",
    "language_difference": "检测语言非英语",
    "asr_uncertain": "识别结果不确定",
    "insufficient_speech": "未获得可靠语音文本",
    "digital_silence": "数字静音，无法核对",
    "moderate_difference": "存在部分文字差异",
    "mostly_agree": "文字基本一致",
}
RULES = {"mostly_agree_max_wer": .15, "moderate_difference_max_wer": .4,
         "alternative_max_distance": .25, "alternative_min_improvement": .3,
         "alternative_min_words": 5}


def normalize_text(text):
    """Small, model-independent comparison rules; no spelling or semantic repair."""
    text = unicodedata.normalize("NFKC", text).replace("’", "'").replace("‘", "'")
    text = re.sub(r"\[[^\]]*\]", " ", text)  # Speaker/stage annotations, not speech.
    text = re.sub(r"\b(?:[A-Za-z]\.)+[A-Za-z]\.?", lambda m: m[0].replace(".", ""), text)
    text = text.casefold()
    text = re.sub(r"\b(?:can't|cannot)\b", "can not", text)
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"\bshan't\b", "shall not", text)
    text = re.sub(r"\bi'm\b", "i am", text)
    text = re.sub(r"\b(he|she|it|that|there|what|who|here)'s\b", r"\1 is", text)
    for suffix, expansion in [("n't", " not"), ("'re", " are"), ("'ve", " have"), ("'ll", " will")]:
        text = re.sub(re.escape(suffix) + r"\b", expansion, text)
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = text.replace("'", "")  # Ambiguous 'd and possessives are not expanded.
    tokens = re.findall(r"[^\W_]+(?:\.\d+)?", text)
    small = dict(zip("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split(), range(20)))
    tens = dict(zip("twenty thirty forty fifty sixty seventy eighty ninety".split(), range(20, 100, 10)))
    ordinals = dict(zip("first second third fourth fifth sixth seventh eighth ninth tenth".split(),
                        "1st 2nd 3rd 4th 5th 6th 7th 8th 9th 10th".split()))
    numbers, i = [], 0
    while i < len(tokens):
        word = tokens[i]
        if word in tens:
            value = tens[word]
            if i + 1 < len(tokens) and 0 < small.get(tokens[i + 1], 0) < 10:
                i += 1
                value += small[tokens[i]]
            numbers.append(str(value))
        else:
            numbers.append(str(small.get(word, ordinals.get(word, word))))
        i += 1
    # Combine scale expressions, while keeping "one one" and "one and two" separate.
    for scale_word, scale in [("hundred", 100), ("thousand", 1000), ("million", 1000000)]:
        combined, i = [], 0
        while i < len(numbers):
            word = numbers[i]
            if word == scale_word:
                base = combined.pop() if combined and (combined[-1].isascii() and combined[-1].isdigit() or combined[-1] == "a") else "1"
                value = (1 if base == "a" else int(base)) * scale
                j = i + 1
                if j < len(numbers) and numbers[j] == "and":
                    j += 1
                if j < len(numbers) and numbers[j].isascii() and numbers[j].isdigit() and int(numbers[j]) < scale:
                    value += int(numbers[j])
                    i = j
                combined.append(str(value))
            else:
                combined.append(word)
            i += 1
        numbers = combined
    return " ".join(numbers)


def edit_counts(reference, hypothesis):
    """Unit-cost Levenshtein alignment, including insertions beyond 100% WER."""
    n, m = len(reference), len(hypothesis)
    costs = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        costs[i][0] = i
    for j in range(m + 1):
        costs[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            costs[i][j] = min(costs[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]),
                              costs[i - 1][j] + 1, costs[i][j - 1] + 1)
    i, j = n, m
    counts = {"substitutions": 0, "deletions": 0, "insertions": 0, "matches": 0}
    while i or j:
        if i and j and reference[i - 1] == hypothesis[j - 1] and costs[i][j] == costs[i - 1][j - 1]:
            counts["matches"] += 1; i -= 1; j -= 1
        elif i and j and costs[i][j] == costs[i - 1][j - 1] + 1:
            counts["substitutions"] += 1; i -= 1; j -= 1
        elif i and costs[i][j] == costs[i - 1][j] + 1:
            counts["deletions"] += 1; i -= 1
        else:
            counts["insertions"] += 1; j -= 1
    return {**counts, "reference_words": n, "asr_words": m,
            "word_edit_distance": costs[n][m], "wer": costs[n][m] / n if n else None,
            "normalized_word_distance": costs[n][m] / max(n, m, 1)}


def classify(record, metrics):
    if record["digital_silence"]:
        return "digital_silence"
    if not record["asr_text"].strip():
        return "insufficient_speech"
    if record["generation_limit"] or not record.get("language"):
        return "asr_uncertain"
    if record["language"] != "en":
        return "language_difference"
    if metrics["wer"] is None:
        return "asr_uncertain"
    if metrics["wer"] <= RULES["mostly_agree_max_wer"]:
        return "mostly_agree"
    if metrics["wer"] <= RULES["moderate_difference_max_wer"]:
        return "moderate_difference"
    return "large_difference"


def alternative_matches(key, hypothesis, references, own_distance):
    if len(hypothesis) < RULES["alternative_min_words"]:
        return []
    candidates = []
    for other, words in references.items():
        if other == key or len(words) < RULES["alternative_min_words"]:
            continue
        score = edit_counts(words, hypothesis)["normalized_word_distance"]
        if score <= RULES["alternative_max_distance"] and own_distance - score >= RULES["alternative_min_improvement"]:
            candidates.append({"key": other, "normalized_word_distance": score})
    return sorted(candidates, key=lambda x: (x["normalized_word_distance"], x["key"]))[:3]


class QwenRecognizer:
    def __init__(self, device, beams):
        import torch
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration
        self.device, self.beams = device, beams
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(QWEN_MODEL["id"], revision=QWEN_MODEL["revision"])
        self.model = Qwen3ASRForConditionalGeneration.from_pretrained(
            QWEN_MODEL["id"], revision=QWEN_MODEL["revision"], dtype=self.dtype,
            attn_implementation="eager").to(device).eval()

    def transcribe(self, waveforms):
        import torch
        from transformers import LogitsProcessor
        from transformers.models.qwen3_asr.processing_qwen3_asr import LANGUAGE_CODE_TO_NAME

        class FiniteScores(LogitsProcessor):
            def __call__(self, input_ids, scores):
                if torch.isnan(scores).any() or torch.isposinf(scores).any() or not torch.isfinite(scores).any(-1).all():
                    raise FloatingPointError("Qwen ASR产生非有限生成分数，停止核查，不保存伪转录")
                return scores

        inputs = self.processor.apply_transcription_request(
            audio=waveforms, language=None, prompt=None, return_tensors="pt",
            processor_kwargs={"sampling_rate": 16000, "padding": True}).to(self.device, self.dtype)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, do_sample=False, num_beams=self.beams,
                                            max_new_tokens=512, return_dict_in_generate=True,
                                            logits_processor=[FiniteScores()])
        tokens = generated.sequences[:, inputs["input_ids"].shape[1]:]
        raw = self.processor.tokenizer.batch_decode(tokens, skip_special_tokens=False)
        decoded = self.processor.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        parsed = self.processor.parse_output(decoded)
        names = {v.lower(): k for k, v in LANGUAGE_CODE_TO_NAME.items()}
        names["hebrew"] = "he"  # Preserve an emitted label; Hebrew is outside documented support.
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        result = []
        for item, raw_text, ids in zip(parsed, raw, tokens.tolist()):
            language = item["language"]
            code = names.get(language.lower(), language.lower()) if language else None
            end = next((i for i, t in enumerate(ids) if t in eos), len(ids))
            result.append({"asr_text": item["transcription"].strip(), "language": code,
                           "generation_limit": end == len(ids), "generated_token_count": end,
                           "raw_asr_output": raw_text, "language_label": language})
        return result


def source_rows(cfg):
    from .pipeline import Cache
    root = Path(cfg["output_dir"])
    manifest = read_json(root / "manifest.json")
    if file_hash(cfg["labels"]) != manifest["label_sha256"]:
        raise ValueError("原始转录表已改变，须先核查正式样本清单")
    cache = Cache(cfg)
    result = []
    for sample in manifest["samples"]:
        expected = (Path(cfg["video_root"]) / sample["video_id"] / (sample["clip_id"] + ".mp4")).resolve()
        if expected != Path(sample["video"]) or file_hash(expected) != sample["video_sha256"]:
            raise ValueError(f"视频身份或内容不符：{sample['key']}")
        if not cache.valid(sample, "media"):
            raise ValueError(f"媒体缓存缺失或过期：{sample['key']}")
        directory = cache.directory(sample)
        media = read_json(directory / "media.json")
        if media["audio_duration"] > 30:
            raise ValueError("当前审计仅适用于不超过30秒的片段，不允许截断长音频")
        align = read_json(directory / "align.json")["words"]
        result.append({**sample, "audio_path": str(directory / "audio.wav"),
                       "audio_sha256": file_hash(directory / "audio.wav"),
                       "audio_seconds": media["audio_duration"], "video_seconds": media["duration"],
                       "digital_silence": media["silent_audio"],
                       "structural_valid_word_ratio": sum(w["time_valid"] for w in align) / len(align)})
    return result


def compare(rows, transcriptions, supported_languages):
    references = {s["key"]: normalize_text(s["raw_text"]).split() for s in rows}
    compared = []
    for sample in rows:
        record = {**sample, **transcriptions[sample["key"]]}
        hypothesis = normalize_text(record["asr_text"]).split()
        metrics = edit_counts(references[sample["key"]], hypothesis)
        status = classify(record, metrics)
        comparable = status in ("mostly_agree", "moderate_difference", "large_difference")
        alternatives = alternative_matches(sample["key"], hypothesis, references,
                                           metrics["normalized_word_distance"]) if comparable else []
        compared.append({**record, **metrics, "status": status, "status_label": STATUS[status],
                         "language_supported": record["language"] in supported_languages if record["language"] else None,
                         "wer_comparable": comparable, "normalized_reference": " ".join(references[sample["key"]]),
                         "normalized_asr": " ".join(hypothesis), "alternative_matches": alternatives})
    return compared


def write_reports(out, metadata, records):
    counts = dict(Counter(r["status"] for r in records))
    model_id = metadata["settings"]["model"]["id"]
    summary = {"samples": len(records), "status_counts": counts,
               "asr_inferred": sum(not r["digital_silence"] for r in records),
               "unsupported_language_labels": [r["key"] for r in records if r["language_supported"] is False],
               "clips_with_alternative_match": sum(bool(r["alternative_matches"]) for r in records)}
    atomic_json(out / "results.json", {"metadata": metadata, "summary": summary, "records": records})
    fields = ["key", "status_label", "language", "language_label", "language_supported",
              "wer", "wer_comparable", "matches", "substitutions", "deletions", "insertions",
              "reference_words", "asr_words", "audio_seconds", "structural_valid_word_ratio",
              "raw_text", "asr_text", "normalized_reference", "normalized_asr", "alternative_matches",
              "video", "video_sha256", "audio_sha256"]
    order = {k: i for i, k in enumerate(STATUS)}
    records = sorted(records, key=lambda r: (order[r["status"]], -(r["wer"] or 0), r["key"]))
    with (out / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in records:
            data = {k: row[k] for k in fields}
            data["alternative_matches"] = "; ".join(x["key"] for x in row["alternative_matches"])
            writer.writerow(data)
    esc = html.escape
    content = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'>",
               "<meta name='viewport' content='width=device-width,initial-scale=1'>",
               "<title>第一问：ASR与题目转录的一致性核查</title>",
               "<style>body{font:16px/1.6 system-ui;margin:28px;color:#163440;background:#f8fafb}h1{font-size:25px}"
               "table{border-collapse:collapse;width:100%;background:white}td,th{padding:12px;border:1px solid #d4e0e6;vertical-align:top}"
               "th{background:#145d6c;color:white;position:sticky;top:0}td:nth-child(2),td:nth-child(3){width:30%}"
               "small{color:#596d77}audio{width:240px}input,select{padding:8px;margin:8px}"
               ".tag{display:inline-block;padding:4px 9px;background:#e0edf1;margin:4px}details{margin-top:10px;overflow-wrap:anywhere}"
               ".count{font-weight:600}a{color:#0c6477}</style>",
               "<h1>视频语音与题目转录：独立ASR自动核查</h1>",
               f"<p>共{len(records)}条；模型 <a href='https://huggingface.co/{esc(model_id)}'>{esc(model_id.split('/')[-1])}</a>。"
               "仅音频进入识别器；自动检测语言，按原语种转写，没有题目文本提示。原始转录、正式特征和掩码均未改动。</p>",
               "<p>这里的WER=(替换+删除+插入)/题目文本规范化词数，衡量两份文本的差异，可超过100%。"
               "题目文本与ASR都可能有误，因此它不是经过真值标定的识别准确率，也不能单独证明视频错配。"
               "非英语、静音及识别不可靠时，WER不用于英语一致性判定。</p>",
               "<p>Qwen输出语言标签，不提供本流程可用的语言概率或无语音概率。模型输出的语言标签若不在处理器支持列表中，"
               "会标记超出支持范围；该标签本身也可能出错，不能当作已确认的语种。</p>"]
    content.append("<div>" + "".join(f"<span class='tag'>{esc(STATUS[k])}：{counts.get(k, 0)}</span>" for k in STATUS) + "</div>")
    content.append("<p>筛查规则：英语且有可用识别文本时，WER≤15%为基本一致，15%—40%为部分差异，>40%为较大差异。"
                   "这些阈值仅用于排序；短句、口音、背景声和口语词都可能造成误报。"
                   "比较采用独立规范化规则q1_en_v1：统一大小写、标点、常见英文缩写与数字表达；保留原文供核对。"
                   "不推断歧义缩写、不纠正拼写，复杂数字读法仍可能产生差异；片段边缘多说几个词也会提高WER。"
                   "数字静音直接记为无法核对，不让模型凭空生成文字。</p>")
    content.append("<label>筛选 <select id='status'><option value=''>全部</option>" +
                   "".join(f"<option value='{k}'>{esc(v)}</option>" for k, v in STATUS.items()) + "</select></label>"
                   "<label>搜索 <input id='search' placeholder='样本ID或文字'></label> <span id='count'></span>"
                   "<p><a href='comparison.csv'>下载全量CSV</a> · <a href='results.json'>模型版本、原始结果与核查记录</a></p>"
                   f"<table><thead><tr><th>样本与音频</th><th>题目原文</th><th>{esc(model_id.split('/')[-1])}</th>"
                   "<th>核查结果</th></tr></thead><tbody>")
    for r in records:
        audio = quote(os.path.relpath(r["audio_path"], out))
        video = quote(os.path.relpath(r["video"], out))
        wer = f"{r['wer']:.1%}" if r["wer_comparable"] else "不适用"
        lang = r["language"] or "无"
        other = "; ".join(x["key"] for x in r["alternative_matches"])
        language_note = "<p><b>模型输出语言标签超出支持范围</b></p>" if r["language_supported"] is False else ""
        content.append(f"<tr data-status='{r['status']}'><td><b>{esc(r['key'])}</b><br>"
                       f"{r['audio_seconds']:.2f}秒 · <a href='{video}'>原视频</a><br>"
                       f"<audio controls preload='none' src='{audio}'></audio><br>"
                       f"<small>结构有效词时间 {r['structural_valid_word_ratio']:.1%}</small></td>"
                       f"<td>{esc(r['raw_text'])}<details><summary>规范化文本</summary>{esc(r['normalized_reference'])}</details></td>"
                       f"<td dir='auto'>{esc(r['asr_text']) or '（未获得转写）'}<details><summary>规范化文本</summary>{esc(r['normalized_asr'])}</details></td>"
                       f"<td><b>{esc(r['status_label'])}</b><br>模型语言标签：{esc(lang)}<br>WER：{wer}<br>"
                       f"<small>替换/删除/插入：{r['substitutions']}/{r['deletions']}/{r['insertions']}</small>"
                       + (f"<p>更相似的其他片段（候选）：{esc(other)}</p>" if other else "") + language_note + "</td></tr>")
    content.append("</tbody></table><script>const rows=[...document.querySelectorAll('tbody tr')];"
                   "function filter(){let n=0;const s=document.querySelector('#status').value,q=document.querySelector('#search').value.toLowerCase();"
                   "for(const r of rows){r.hidden=!!((s&&r.dataset.status!==s)||!r.textContent.toLowerCase().includes(q));if(!r.hidden)n++;}"
                   "document.querySelector('#count').textContent=`显示 ${n} / ${rows.length} 条`;}"
                   "document.querySelector('#status').addEventListener('change',filter);document.querySelector('#search').addEventListener('input',filter);filter();</script></html>")
    (out / "report.html").write_text("\n".join(content), encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--beams", type=int, default=1, help="默认1：贪心解码")
    parser.add_argument("--device")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.beams < 1 or (args.limit is not None and args.limit < 1):
        parser.error("batch-size、beams、limit必须为正")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg = load_config(args.config)
    out = Path(cfg["output_dir"]) / "asr_audit"
    out.mkdir(exist_ok=True)
    with (out / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = source_rows(cfg)
        if args.limit:
            rows = rows[:args.limit]
        import torch
        from transformers.models.qwen3_asr.processing_qwen3_asr import LANGUAGE_CODE_TO_NAME
        torch.set_num_threads(4)
        torch.manual_seed(2026)
        device = args.device or cfg["device"]
        settings = {"model": QWEN_MODEL, "device": device, "dtype": "float16" if device.startswith("cuda") else "float32",
                    "attention": "eager", "batch_size": args.batch_size, "num_beams": args.beams,
                    "max_new_tokens": 512, "task": "transcribe", "language": "auto",
                    "reference_prompt": False, "condition_on_previous_clip": False, "seed": 2026,
                    "inference_code_sha256": digest(inspect.getsource(QwenRecognizer)),
                    "transformers": importlib.metadata.version("transformers"), "torch": torch.__version__}
        signature = digest(settings)
        prior = read_json(out / "results.json") if (out / "results.json").exists() and not args.force else {}
        cached = {r["key"]: r for r in prior.get("records", [])}
        records, pending = {}, []
        for r in rows:
            old = cached.get(r["key"], {})
            if old.get("inference_signature") == signature and old.get("audio_sha256") == r["audio_sha256"]:
                records[r["key"]] = {k: old[k] for k in ["asr_text", "language",
                    "generation_limit", "generated_token_count", "inference_signature",
                    "raw_asr_output", "language_label"] if k in old}
            elif r["digital_silence"]:
                records[r["key"]] = {"asr_text": "", "language": None, "language_label": None,
                    "raw_asr_output": "", "generation_limit": False, "generated_token_count": 0,
                    "inference_signature": signature}
            else:
                pending.append(r)
        logging.info("核查%d条，缓存复用%d条，数字静音%d条，待识别%d条", len(rows),
                     len(records) - sum(r["digital_silence"] for r in rows),
                     sum(r["digital_silence"] for r in rows), len(pending))
        started = time.monotonic()
        recognizer = QwenRecognizer(device, args.beams) if pending else None
        metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "settings": settings,
                    "classification_rules": RULES, "code_sha256": file_hash(__file__),
                    "manifest_sha256": file_hash(Path(cfg["output_dir"]) / "manifest.json"),
                    "labels_sha256": file_hash(cfg["labels"]), "formal_outputs_modified": False,
                    "normalizer": {"name": "q1_en_v1", "code_sha256": digest(inspect.getsource(normalize_text)),
                                   "rules": "NFKC, casefold, punctuation, bracket annotations, dotted acronyms, common contractions and integer expressions; no spelling repair"},
                    "supported_language_codes": sorted(LANGUAGE_CODE_TO_NAME),
                    "diagnostics_note": "Qwen emits a language label, not calibrated language/no-speech probabilities. Unsupported emitted labels remain unverified.",
                    "interpretation": "ASR-reference disagreement is not an accuracy estimate or confirmed mispairing."}
        if device.startswith("cuda"):
            metadata["gpu"] = torch.cuda.get_device_name(torch.device(device))
            torch.cuda.reset_peak_memory_stats(device)
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            waveforms = []
            for r in batch:
                wave, sr = sf.read(r["audio_path"], dtype="float32")
                if sr != 16000 or wave.ndim != 1 or not np.isfinite(wave).all() or len(wave) > 480000:
                    raise ValueError(f"ASR音频格式无效：{r['key']}")
                waveforms.append(wave)
            predictions = recognizer.transcribe(waveforms)
            if len(predictions) != len(batch):
                raise ValueError("ASR返回条数与输入不一致")
            for row, pred in zip(batch, predictions):
                records[row["key"]] = {**pred, "inference_signature": signature}
                logging.info("%s language=%s text=%s", row["key"], pred["language"], pred["asr_text"])
            completed = [r for r in rows if r["key"] in records]
            metadata["complete"] = len(completed) == len(rows)
            metadata["expected"] = len(rows)
            metadata["elapsed_seconds"] = time.monotonic() - started
            compared = compare(completed, records, LANGUAGE_CODE_TO_NAME)
            # Save progress after each batch; incomplete output is explicitly marked.
            atomic_json(out / "results.json", {"metadata": metadata, "records": compared})
            logging.info("已完成 %d/%d", len(completed), len(rows))
        metadata["complete"], metadata["expected"] = True, len(rows)
        metadata["elapsed_seconds"] = time.monotonic() - started
        metadata["new_asr_inferences"] = len(pending)
        if device.startswith("cuda"):
            metadata["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        compared = compare(rows, records, LANGUAGE_CODE_TO_NAME)
        summary = write_reports(out, metadata, compared)
        logging.info("核查完成：%s", summary)
        print(out / "report.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
