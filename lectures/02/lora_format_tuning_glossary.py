# -*- coding: utf-8 -*-
# description: LoRA format tuning using glossary dataset

# section: 0
# description: Check environment and install libraries
import subprocess
try:
    gpu_info = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout
except FileNotFoundError:
    gpu_info = ""
print(gpu_info or "GPU not found. Go to Runtime -> Change runtime type -> Select T4/A100 GPU.")

PINS = {
    "transformers": "5.18.0",
    "peft": "0.21.1",
    "trl": "1.14.2",
    "accelerate": "1.15.0",
    "datasets": "5.1.0",
}
spec = " ".join(f'"{k}=={v}"' for k, v in PINS.items())

import transformers, peft, trl, accelerate, datasets, torch
print({m.__name__: m.__version__ for m in [transformers, peft, trl, accelerate, datasets, torch]})

# section: 1
# description: Configuration
import datetime
from zoneinfo import ZoneInfo
print(f"[{datetime.datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M:%S')} JST] Executing Section 1 (v2.0.1)...")

from pathlib import Path
import os, json, time

CONFIG = {
    "base_model": "Qwen/Qwen3.5-0.8B",
    "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "seed": 42,
    "num_train_epochs": 20,
    "warmup_steps": 6,
    "lr": 2e-4,
    "lora_r": 16,
    "lora_alpha": 32,
    "max_seq_len": 512,
    "max_new_tokens": 96,
    "gen": {"do_sample": False},
    "work_dir": "/content/qlora_lab_advanced",
    "drive_dir": "/content/drive/MyDrive/qlora_lab_advanced",
}
WORK = Path(CONFIG["work_dir"]); WORK.mkdir(parents=True, exist_ok=True)

USE_DRIVE = True
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    Path(CONFIG["drive_dir"]).mkdir(parents=True, exist_ok=True)
print(json.dumps(CONFIG, ensure_ascii=False, indent=2))

# section: 2
# description: Generate SFT dataset from glossary
import datetime
from zoneinfo import ZoneInfo
print(f"[{datetime.datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M:%S')} JST] Executing Section 2 (v2.0.0)...")

import json
import random
import shutil

# 1. 用語集データの準備 (glossary.json の主要部分を抜粋)
GLOSSARY_RAW = """
{
  "terms": [
    {"term": "AIエンジニアリング", "definition": "学習済みの基盤モデルを部品として使い、実際に役立つアプリケーションやシステムを設計・構築・評価・運用する技術領域。"},
    {"term": "評価と改善のサイクル", "definition": "作る→測る→直すを繰り返す開発の進め方。"},
    {"term": "LLM（大規模言語モデル）", "definition": "大量のテキストで学習し、入力に続くトークンを予測することで文章を生成するモデル。"},
    {"term": "トークン", "definition": "LLMがテキストを扱う最小単位。"},
    {"term": "コンテキストウィンドウ", "definition": "LLMが1回の呼び出しで扱える入力と出力の合計トークン数の上限。"},
    {"term": "プロンプト", "definition": "LLMに渡す入力文。役割、制約、出力形式、例などを含めることで出力を方向づけられる。"},
    {"term": "システムプロンプト", "definition": "利用者の質問とは別に、モデルの振る舞いを指定する入力。"},
    {"term": "ハルシネーション", "definition": "LLMが事実と異なる内容や存在しない情報を、もっともらしく生成してしまう現象。"},
    {"term": "ファインチューニング", "definition": "事前学習済みモデルを、特定のタスクや形式に合ったデータで追加学習させること。"},
    {"term": "SFT", "definition": "入力と望ましい出力のペアを使ってモデルを追加学習させる方法。"},
    {"term": "LoRA", "definition": "事前学習済みの重みは固定したまま、重みの更新分を低ランク行列で表し、その小さな行列だけを学習するPEFT手法。"},
    {"term": "AIエージェント", "definition": "LLMが自分で判断してツールを呼び出し、その結果を見てさらに次の行動を決めるループで目的を達成するシステム。"},
    {"term": "ツール（Tool use / Function calling）", "definition": "モデルが呼び出せるように用意した関数。"},
    {"term": "MCP（Model Context Protocol）", "definition": "AIアプリと外部のツールやデータソースをつなぐためのオープンな標準プロトコル。"}
  ]
}
"""
glossary_data = json.loads(GLOSSARY_RAW)["terms"]
(WORK / "glossary.json").write_text(GLOSSARY_RAW, encoding="utf-8")

# 2. 質問テンプレートと言い換え
TEMPLATES = {
    "train": ["「{term}」とは何ですか？", "{term}について簡単に説明してください。"],
    "valid": ["{term}の意味を教えてください。"],
    "test": ["{term}ってどういうことですか？", "{term}の定義を教えて。"]
}
# 用語集に無い語（根拠なしの確認用）
UNGROUNDED_TERMS = ["RAG", "Transformer", "LangChain", "ディープラーニング", "Python"]
UNGROUNDED_ANSWER = "この用語集には記載がありません。運営に確認が必要です。"

train, valid, test, gold = [], [], [], {}

def generate_faq_context(target_term, all_terms, include_target=True):
    """トークン数上限を超えないよう、対象用語＋ランダムな用語数個を混ぜたダミーのFAQコンテキストを生成する"""
    random.seed(len(target_term)) # 再現性のため
    others = [t for t in all_terms if t["term"] != target_term]
    selected = random.sample(others, min(3, len(others)))
    if include_target:
        target_obj = next(t for t in all_terms if t["term"] == target_term)
        selected.append(target_obj)
    else:
        selected.append(random.choice(others))
    random.shuffle(selected)
    return "講座の用語集からの抜粋:\n" + "\n".join(f"- {t['term']}: {t['definition']}" for t in selected)

def make_example(question, answer, needs_review, context_text):
    target = json.dumps({"answer": answer, "needs_review": needs_review}, ensure_ascii=False)
    return {"question": question, "faq": context_text, "target": target, "needs_review": needs_review}

# 3. データセットの構築
for t in glossary_data:
    term_name = t["term"]
    definition = t["definition"]
    faq_text = generate_faq_context(term_name, glossary_data, include_target=True)

    # 根拠あり
    for q_tpl in TEMPLATES["train"]:
        train.append(make_example(q_tpl.format(term=term_name), definition, False, faq_text))
    for q_tpl in TEMPLATES["valid"]:
        valid.append(make_example(q_tpl.format(term=term_name), definition, False, faq_text))
    for q_tpl in TEMPLATES["test"]:
        q = q_tpl.format(term=term_name)
        test.append(make_example(q, definition, False, faq_text))
        gold[q] = {"needs_review": False, "must_include": [definition[:10]]} # 正解判定に定義の先頭10文字を使用

# 根拠なし (未記載用語)
for i, term in enumerate(UNGROUNDED_TERMS):
    faq_text = generate_faq_context(term, glossary_data, include_target=False)
    q = f"「{term}」とは何ですか？"
    ex = make_example(q, UNGROUNDED_ANSWER, True, faq_text)
    bucket = [train, train, valid, test][i % 4]
    bucket.append(ex)
    if bucket is test:
        gold[q] = {"needs_review": True, "must_include": ["確認"]}

random.seed(CONFIG["seed"])
random.shuffle(train)

# 4. JSONL ファイルの保存 (make_sft_dataset.py の要件)
SYSTEM_PROMPT = "あなたは講座の用語集の回答係です。与えられた用語集だけを根拠に、必ず JSON 1 行で答えてください。形式: {\"answer\": 文字列, \"needs_review\": 真偽値}。用語集に記載が無いことは answer に『確認が必要』と書き、needs_review を true にします。"

def save_jsonl(filename, data_list):
    path = WORK / filename
    with open(path, "w", encoding="utf-8") as f:
        for ex in data_list:
            record = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"用語集:\n{ex['faq']}\n\n質問: {ex['question']}"},
                    {"role": "assistant", "content": ex["target"]}
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path

train_file = save_jsonl("glossary_sft_train.jsonl", train)
valid_file = save_jsonl("glossary_sft_valid.jsonl", valid)
save_jsonl("glossary_sft_test.jsonl", test)
_ = (WORK / "gold.json").write_text(json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8")

# ドライブへコピー
if USE_DRIVE:
    for f in ["glossary_sft_train.jsonl", "glossary_sft_valid.jsonl", "glossary_sft_test.jsonl", "gold.json", "glossary.json"]:
        shutil.copy(WORK / f, Path(CONFIG["drive_dir"]) / f)
    print(f"Drive にデータセットを保存しました: {CONFIG['drive_dir']}")

print(f"生成完了: train={len(train)} valid={len(valid)} test={len(test)} gold={len(gold)}")

# section: 3
# description: Load model and run pre-training evaluation
import datetime
from zoneinfo import ZoneInfo
print(f"[{datetime.datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M:%S')} JST] Executing Section 3 (v2.0.0)...")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import pandas as pd
pd.set_option("display.max_colwidth", None)

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(CONFIG["base_model"], revision=CONFIG["revision"])
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(CONFIG["base_model"], revision=CONFIG["revision"], dtype=torch.float32).to(DEVICE)
model.config.use_cache = True
BASE_MEMORY_GB = round(model.get_memory_footprint() / 1e9, 2)
print("loaded:", CONFIG["base_model"], "| device:", DEVICE, "| 重みのメモリ(GB):", BASE_MEMORY_GB)

SYSTEM = "あなたは講座の用語集の回答係です。与えられた用語集だけを根拠に、必ず JSON 1 行で答えてください。形式: {\"answer\": 文字列, \"needs_review\": 真偽値}。用語集に記載が無いことは answer に『確認が必要』と書き、needs_review を true にします。"

def build_messages(example):
    user = f"用語集:\n{example['faq']}\n\n質問: {example['question']}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]

def generate(model, example):
    prompt = tok.apply_chat_template(build_messages(example), tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=CONFIG["max_new_tokens"], pad_token_id=tok.pad_token_id, **CONFIG["gen"])
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def check_format(text: str):
    try:
        obj = json.loads(text)
    except Exception as e:
        return False, f"json_error: {type(e).__name__}", None
    if not isinstance(obj, dict): return False, "not_object", None
    if set(obj) - {"answer", "needs_review"}: return False, "extra_keys", obj
    if "answer" not in obj or "needs_review" not in obj: return False, "missing_key", obj
    if not isinstance(obj["answer"], str) or not isinstance(obj["needs_review"], bool): return False, "wrong_type", obj
    return True, "", obj

def check_content(obj, gold_entry):
    if obj is None: return False
    if obj.get("needs_review") != gold_entry["needs_review"]: return False
    return all(w in obj.get("answer", "") for w in gold_entry["must_include"])

def mark(ok: bool) -> str:
    return "○" if ok else "×"

def evaluate(model, label: str):
    rows = []
    import time
    t0 = time.time()
    for ex in test:
        raw = generate(model, ex)
        ok_fmt, reason, obj = check_format(raw)
        loose = obj
        if not ok_fmt:
            m = re.search(r"\{.*\}", raw, re.S)
            try: loose = json.loads(m.group(0)) if m else None
            except Exception: loose = None
        ok_content = check_content(loose if isinstance(loose, dict) else None, gold[ex["question"]])
        rows.append({"question": ex["question"], "raw": raw, "format_ok": ok_fmt, "format_reason": reason,
                     "content_ok": ok_content, "both_ok": ok_fmt and ok_content, "gold_needs_review": gold[ex["question"]]["needs_review"]})
    n = len(rows)
    summary = {"label": label, "n": n,
               "format_rate": sum(r["format_ok"] for r in rows)/n,
               "content_rate": sum(r["content_ok"] for r in rows)/n,
               "both_rate": sum(r["both_ok"] for r in rows)/n,
               "seconds": round(time.time()-t0, 1)}
    (WORK / f"eval_{label}.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return summary, rows

before_summary, before_rows = evaluate(model, "before")
display(pd.DataFrame([{"質問": r["question"], "正解の needs_review": r["gold_needs_review"], "学習前の出力": r["raw"],
                       "形式": mark(r["format_ok"]), "形式の失敗理由": r["format_reason"], "内容": mark(r["content_ok"])} for r in before_rows]))

# section: 4
# description: LoRA Fine-tuning
import datetime
from zoneinfo import ZoneInfo
print(f"[{datetime.datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M:%S')} JST] Executing Section 4 (v2.0.0)...")

from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import time

train_ds = load_dataset('json', data_files=str(WORK / "glossary_sft_train.jsonl"))["train"]
valid_ds = load_dataset('json', data_files=str(WORK / "glossary_sft_valid.jsonl"))["train"]

model.config.use_cache = False
lora = LoraConfig(r=CONFIG["lora_r"], lora_alpha=CONFIG["lora_alpha"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
USE_AMP = False
sft = SFTConfig(output_dir=str(WORK / "ckpt"), num_train_epochs=CONFIG["num_train_epochs"], per_device_train_batch_size=4, gradient_accumulation_steps=2,
                learning_rate=CONFIG["lr"], lr_scheduler_type="cosine", warmup_steps=CONFIG["warmup_steps"], logging_steps=10, save_steps=10, save_total_limit=2,
                fp16=USE_AMP, max_length=CONFIG["max_seq_len"], packing=False, report_to="none", seed=CONFIG["seed"])
trainer = SFTTrainer(model=model, args=sft, train_dataset=train_ds, eval_dataset=valid_ds, peft_config=lora, processing_class=tok)
trainer.model.print_trainable_parameters()
t0 = time.time()
trainer.train()
print(f"学習時間: {time.time()-t0:.0f} 秒 | ステップ数: {trainer.state.global_step} | エポック: {trainer.state.epoch:.1f}")
ADAPTER_DIR = WORK / "adapter"
trainer.model.save_pretrained(ADAPTER_DIR)
if USE_DRIVE:
    import shutil; shutil.copytree(ADAPTER_DIR, Path(CONFIG["drive_dir"]) / "adapter", dirs_exist_ok=True)
tuned = trainer.model
tuned.config.use_cache = True
_ = tuned.eval()

# section: 5
# description: Post-training evaluation
import datetime
from zoneinfo import ZoneInfo
print(f"[{datetime.datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M:%S')} JST] Executing Section 5 (v2.0.0)...")

after_summary, after_rows = evaluate(tuned, "after")
compare = pd.DataFrame([before_summary, after_summary]).set_index("label")[["n", "format_rate", "content_rate", "both_rate", "seconds"]]
display(compare)

# Markdown形式でテキストとして出力する
print("\n--- 以下のMarkdownテキストをコピーしてください ---\n")
print(compare.to_markdown(index=False))

regressed = [b["question"] for b, a in zip(before_rows, after_rows) if b["both_ok"] and not a["both_ok"]]
print("退行:", regressed)

detail = pd.DataFrame([{"質問": b["question"], "正解の needs_review": b["gold_needs_review"],
                        "学習前の出力": b["raw"], "学習前 形式/内容": f"{mark(b['format_ok'])}/{mark(b['content_ok'])}",
                        "学習後の出力": a["raw"], "学習後 形式/内容": f"{mark(a['format_ok'])}/{mark(a['content_ok'])}"}
                       for b, a in zip(before_rows, after_rows)])
display(detail)

# Markdown形式でテキストとして出力する
print("\n--- 以下のMarkdownテキストをコピーしてください ---\n")
print(detail.to_markdown(index=False))

# section: 6
# description: Knowledge-based questions
import datetime
from zoneinfo import ZoneInfo
print(f"[{datetime.datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M:%S')} JST] Executing Section 6 (v2.0.0)...")

KNOWLEDGE_QUESTIONS = [
    "「RAG」とは何ですか？",                        # 用語集に無い
    "「LangChain」について教えてください",            # 用語集に無い
    "「ディープラーニング」の定義を教えてください",        # 用語集に無い
    "「ファインチューニング」とは何ですか？",           # 用語集にある（対照）
]

def classify(raw):
    ok, _, obj = check_format(raw)
    if not ok or obj is None: return "format_error"
    return "needs_review" if obj["needs_review"] else "asserted"

knowledge_rows = []
for q in KNOWLEDGE_QUESTIONS:
    # 知識評価用なので、faqの文脈は空文字列にしてモデル自身の知識（または学習内容）のみで答えさせる
    ex = {"question": q, "faq": "", "target": "", "needs_review": True}
    with tuned.disable_adapter():
        before_raw = generate(tuned, ex)
    after_raw = generate(tuned, ex)
    knowledge_rows.append({"質問": q, "FAQ にある": q in gold, "学習前の分類": classify(before_raw), "学習前の出力": before_raw,
                 "学習後の分類": classify(after_raw), "学習後の出力": after_raw})
detail = pd.DataFrame(knowledge_rows)
display(detail)

# Markdown形式でテキストとして出力する
print("\n--- 以下のMarkdownテキストをコピーしてください ---\n")
print(detail.to_markdown(index=False))