"""
Cross-sectional smoke test for the Tripod framework.

What it does:
- Builds a tiny local "vector DB" of heuristics (sentence-transformer embeddings).
- Downloads a small chatty LM (TinyLlama chat 1.1B or distilgpt2 fallback on CPU) and fine-tunes on dummy data (80/20 split).
- Uses retrieved heuristics + a role/system prompt to generate actions for the held-out 20%.
- Reports simple string-match accuracy and emits predictions to stdout.

Note: This script downloads models from Hugging Face on first run (internet required).
"""

from __future__ import annotations

import json
import logging
import random
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

os.environ["PYTORCH_MPS_ENABLED"] = "0"  # Force CPU to avoid MPS OOM on macOS.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SmokeTest")

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DIR = BASE_DIR / "training_data" / "vectordb"
VECTOR_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cpu")  # Keep smoke test consistent across machines (avoid MPS quirks).
BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CPU_FALLBACK_MODEL_ID = "sshleifer/tiny-gpt2"
USE_4BIT = True  # Attempt 4-bit when CUDA + bitsandbytes are available; fallback otherwise.
GEN_MAX_NEW_TOKENS = 24
GEN_NUM_BEAMS = 1
GEN_DO_SAMPLE = False
GEN_REP_PENALTY = 1.2

SYSTEM_PROMPT = (
    "You are a safety-first Industrial IoT controller. "
    "Use retrieved heuristics before deciding. "
    "Respond ONLY with actions in the schema: "
    "increase_fan: <int>; reduce_voltage: <float>; schedule_maintenance: <true|false>; "
    "maintain_settings: <true|false>; fan_speed: <int>; keep_voltage: <float>. "
    "Do not add extra words."
)

HEURISTICS = [
    "If temp >= 90C or status is critical, drop voltage by at least 0.2 and set fan to 90+.",
    "If vibration > 1.0 regardless of temp, schedule maintenance and reduce voltage slightly.",
    "If status is stable and temp < 70C, maintain settings and keep fan near 50-60.",
    "If status is warning and temp >= 80C, raise fan to 75-85 and reduce voltage 0.05-0.1.",
    "If vibration is between 0.5 and 1.0 with warning, set fan near 65-75 and plan maintenance.",
]

DUMMY_SAMPLES: List[Dict] = [
    {"sensor": {"temp": 92, "vibration": 0.4, "status": "critical"}, "action": "increase_fan: 90; reduce_voltage: 0.2"},
    {"sensor": {"temp": 86, "vibration": 0.3, "status": "warning"}, "action": "increase_fan: 80; reduce_voltage: 0.1"},
    {"sensor": {"temp": 72, "vibration": 0.5, "status": "warning"}, "action": "increase_fan: 70; keep_voltage: 0.0"},
    {"sensor": {"temp": 65, "vibration": 1.2, "status": "warning"}, "action": "schedule_maintenance: true; reduce_voltage: 0.05"},
    {"sensor": {"temp": 78, "vibration": 0.2, "status": "stable"}, "action": "maintain_settings: true; fan_speed: 60"},
    {"sensor": {"temp": 95, "vibration": 0.6, "status": "critical"}, "action": "increase_fan: 95; reduce_voltage: 0.25"},
    {"sensor": {"temp": 60, "vibration": 0.1, "status": "stable"}, "action": "maintain_settings: true; fan_speed: 50"},
    {"sensor": {"temp": 82, "vibration": 0.9, "status": "warning"}, "action": "increase_fan: 75; reduce_voltage: 0.08"},
    {"sensor": {"temp": 88, "vibration": 0.2, "status": "critical"}, "action": "increase_fan: 85; reduce_voltage: 0.15"},
    {"sensor": {"temp": 70, "vibration": 0.8, "status": "warning"}, "action": "increase_fan: 65; schedule_maintenance: true"},
]

AUG_FACTOR = 2  # Replicate dummy samples; keep small for CPU fallback.
MAX_CORPUS = 40  # Cap total samples to keep smoke test tractable.

FEW_SHOTS = [
    {"sensor": {"temp": 92, "vibration": 0.4, "status": "critical"}, "action": "increase_fan: 90; reduce_voltage: 0.2"},
    {"sensor": {"temp": 65, "vibration": 1.2, "status": "warning"}, "action": "schedule_maintenance: true; reduce_voltage: 0.05"},
]


def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def describe_sensor(sensor: Dict) -> str:
    return f"temp={sensor['temp']}C | vibration={sensor['vibration']}g | status={sensor['status']}"


@dataclass
class RetrievedContext:
    text: str
    score: float
    metadata: Dict


class LocalVectorDB:
    """Lightweight in-memory vector store with persistence of metadata/heuristics."""

    def __init__(self, heuristics: List[str]):
        self.heuristics = heuristics
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(DEVICE))
        self.embeddings = self.embedder.encode(heuristics, normalize_embeddings=True)

    def search(self, query: str, top_k: int = 1) -> List[RetrievedContext]:
        query_vec = self.embedder.encode([query], normalize_embeddings=True)[0]
        scores = np.dot(self.embeddings, query_vec)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedContext(text=self.heuristics[i], score=float(scores[i]), metadata={"rule_id": i})
            for i in top_idx
        ]

    def persist(self, path: Path):
        payload = [
            {"text": h, "metadata": {"rule_id": idx}}
            for idx, h in enumerate(self.heuristics)
        ]
        path.write_text(json.dumps(payload, indent=2))
        logger.info("Persisted heuristic store to %s", path)


def build_prompt(sensor: Dict, rag_context: str) -> str:
    few_shot_block = "\n".join(
        [
            f"EXAMPLE SENSOR: {json.dumps(fs['sensor'])}\nEXAMPLE ACTION: {fs['action']}"
            for fs in FEW_SHOTS
        ]
    )
    return (
        f"{SYSTEM_PROMPT}\n"
        f"{few_shot_block}\n"
        f"HEURISTICS:\n{rag_context}\n"
        f"SENSOR:\n{json.dumps(sensor)}\n"
        f"ACTION:"
    )


def prepare_datasets(db: LocalVectorDB):
    set_seeds(42)
    all_samples: List[Dict] = []
    for _ in range(AUG_FACTOR):
        for sample in DUMMY_SAMPLES:
            all_samples.append({"sensor": sample["sensor"].copy(), "action": sample["action"]})

    if len(all_samples) > MAX_CORPUS:
        all_samples = all_samples[:MAX_CORPUS]

    random.shuffle(all_samples)
    split_idx = max(1, int(len(all_samples) * 0.8))
    train_samples = all_samples[:split_idx]
    test_samples = all_samples[split_idx:]

    def to_text(sample, include_action: bool):
        rag_hit = db.search(describe_sensor(sample["sensor"]), top_k=1)[0]
        prompt = build_prompt(sample["sensor"], rag_hit.text)
        return f"{prompt} {sample['action']}" if include_action else prompt

    train_texts = [to_text(s, include_action=True) for s in train_samples]
    test_prompts = [to_text(s, include_action=False) for s in test_samples]
    return train_samples, test_samples, Dataset.from_dict({"text": train_texts}), Dataset.from_dict({"text": test_prompts})


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    def _tokenize(batch):
        tokens = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=256,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    return dataset.map(_tokenize, batched=True, remove_columns=["text"])


def train_model(train_ds: Dataset, tokenizer: AutoTokenizer):
    effective_model_id = BASE_MODEL_ID
    per_device_bs = 4
    num_epochs = 2
    load_kwargs = {}
    use_cuda_4bit = USE_4BIT and torch.cuda.is_available()
    if use_cuda_4bit:
        try:
            import bitsandbytes as _bnb  # noqa: F401

            load_kwargs.update(
                {
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": torch.float16,
                    "bnb_4bit_use_double_quant": True,
                    "device_map": "auto",
                }
            )
            logger.info("Loading model in 4-bit (requires CUDA + bitsandbytes).")
        except ImportError:
            logger.warning("bitsandbytes not available; falling back to fp16 on CPU.")
    else:
        effective_model_id = CPU_FALLBACK_MODEL_ID  # Smaller fallback for CPU-only environments.
        per_device_bs = 1
        num_epochs = 1
        logger.info("CUDA unavailable; using fallback model %s on CPU to avoid OOM.", effective_model_id)

    model = AutoModelForCausalLM.from_pretrained(
        effective_model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        **load_kwargs,
    )
    if "device_map" not in load_kwargs:
        model = model.to(DEVICE)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.eos_token_id

    args = TrainingArguments(
        output_dir=str(BASE_DIR / "training_data" / "smoke_run"),
        per_device_train_batch_size=per_device_bs,
        num_train_epochs=num_epochs,
        logging_steps=50,
        save_strategy="no",
        learning_rate=5e-4,
        weight_decay=0.0,
        report_to="none",
        use_mps_device=False,
        no_cuda=not torch.cuda.is_available(),
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_ds,
    )
    logger.info("Starting tiny model fine-tune on %s samples", len(train_ds))
    trainer.train()
    return model


def evaluate(model, tokenizer: AutoTokenizer, db: LocalVectorDB, test_samples: List[Dict]):
    logger.info("Evaluating on %s held-out samples", len(test_samples))
    model = model.to(DEVICE)
    hits = 0
    results = []
    for sample in test_samples:
        rag_hit = db.search(describe_sensor(sample["sensor"]), top_k=1)[0]
        prompt = build_prompt(sample["sensor"], rag_hit.text)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            do_sample=GEN_DO_SAMPLE,
            num_beams=GEN_NUM_BEAMS,
            repetition_penalty=GEN_REP_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        expected = sample["action"]
        # Loose match: check if primary verb from expected is present.
        matched = expected.split(";")[0].split(":")[0].strip().lower() in generated.lower()
        hits += int(matched)
        results.append(
            {
                "prompt": prompt,
                "generated": generated,
                "expected": expected,
                "match": matched,
                "retrieved": rag_hit.text,
                "retrieval_score": rag_hit.score,
            }
        )

    accuracy = hits / max(1, len(test_samples))
    return accuracy, results


def main():
    set_seeds(42)
    db = LocalVectorDB(HEURISTICS)
    db.persist(VECTOR_DIR / "heuristics.json")

    train_samples, test_samples, train_ds, test_prompts = prepare_datasets(db)

    tokenizer_id = BASE_MODEL_ID if torch.cuda.is_available() else CPU_FALLBACK_MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized_train = tokenize_dataset(train_ds, tokenizer)

    model = train_model(tokenized_train, tokenizer)

    accuracy, results = evaluate(model, tokenizer, db, test_samples)

    logger.info("Smoke test accuracy: %.2f", accuracy)
    for idx, res in enumerate(results, 1):
        print(f"\n[Sample {idx}] match={res['match']}")
        print("Prompt:\n", res["prompt"])
        print("Expected:", res["expected"])
        print("Generated:", res["generated"])
        print("Retrieved:", res["retrieved"])
        print("Retrieval score:", round(res["retrieval_score"], 3))


if __name__ == "__main__":
    main()
