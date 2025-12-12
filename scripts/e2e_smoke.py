"""
End-to-end smoke test: Training (LoRA/QLoRA) + RAG + Prompting + Evaluation.

Designed to run on a GPU training node (e.g., RunPod).

Flow:
1) Generate synthetic dataset + RAG docs.
2) Ingest RAG docs into local vector store.
3) Build a training file that includes retrieved RAG context for each example.
4) Train a LoRA adapter on TinyLlama (4-bit if CUDA).
5) Evaluate on the held-out set with two modes:
   - with RAG context in the prompt
   - without RAG context (ablation)
"""

from __future__ import annotations

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.rag import RAGLeg
from core.training import TrainingLeg
from core.prompting import PromptLeg
from core.config import TripodConfig


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("E2ESmoke")


SMOKE_DIR = ROOT / "training_data" / "smoke"
SMOKE_CONFIG = ROOT / "configs" / "smoke_config.yaml"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000, help="Total synthetic samples to generate")
    p.add_argument("--eval-samples", type=int, default=200, help="Max evaluation samples (0 = all)")
    return p.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def build_query(example: Dict[str, Any]) -> str:
    s = example["sensor_data"]
    return f"policy_id={example['policy_id']} temp={s['temp']} vib={s['vibration']} status={s['status']}"


def parse_action(generated: str) -> Tuple[str, Dict[str, Any]]:
    # Try to parse JSON in the output; fall back to empty.
    try:
        start = generated.find("{")
        end = generated.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(generated[start : end + 1])
            return str(obj.get("action", "")), obj.get("parameters", {}) or {}
    except Exception:
        pass
    return "", {}


def _build_rag_context(rag: RAGLeg, row: Dict[str, Any]) -> str:
    policy_ctx = rag.run(query=build_query(row), filters={"policy_id": row["policy_id"]})
    heur_ctx = rag.run(query=build_query(row), filters={"type": "heuristic"})
    return "\n".join([c for c in [policy_ctx, heur_ctx] if c]).strip()


def _num(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def load_llm(model_id: str, adapter_dir: Optional[Path]):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = base
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, str(adapter_dir))

    model.eval()
    return model, tokenizer


def evaluate(
    model,
    tokenizer,
    prompter: PromptLeg,
    rag: RAGLeg,
    test_rows: List[Dict[str, Any]],
    use_rag: bool,
    max_samples: int = 0,
) -> Dict[str, float]:
    import torch

    if max_samples and max_samples > 0:
        test_rows = test_rows[:max_samples]

    action_hits = 0
    param_hits = 0
    thermal_param_hits = 0
    thermal_total = 0

    for row in test_rows:
        rag_context = _build_rag_context(rag, row) if use_rag else ""

        prompt = prompter.run(
            {
                "domain": row["domain"],
                "rag_context": rag_context,
                "sensor_data": {"policy_id": row["policy_id"], **row["sensor_data"]},
            }
        )

        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        out = model.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        action, params = parse_action(gen)

        expected = row["expected"]
        expected_action = expected["action"]
        expected_params = expected["parameters"]

        action_ok = action == expected_action
        if action_ok:
            action_hits += 1

        # Parameter match (only counted when action matches)
        params_ok = False
        if action_ok and isinstance(params, dict):
            fan_pred = _num(params.get("fan_speed"))
            vb_pred = _num(params.get("voltage_bias"))
            fan_exp = _num(expected_params.get("fan_speed"))
            vb_exp = _num(expected_params.get("voltage_bias"))
            if fan_pred is not None and vb_pred is not None and fan_exp is not None and vb_exp is not None:
                fan_ok = int(round(fan_pred)) == int(round(fan_exp))
                vb_ok = abs(vb_pred - vb_exp) <= 0.02
                params_ok = fan_ok and vb_ok
        if params_ok:
            param_hits += 1

        if expected_action == "set_thermal_profile":
            thermal_total += 1
            if params_ok:
                thermal_param_hits += 1

    n = max(1, len(test_rows))
    return {
        "action_acc": action_hits / n,
        "param_acc": param_hits / n,
        "thermal_param_acc": thermal_param_hits / max(1, thermal_total),
    }


def main():
    args = parse_args()
    # 1) Generate dataset/docs
    import subprocess

    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            "python",
            str(ROOT / "scripts" / "generate_smoke_dataset.py"),
            "--out-dir",
            str(SMOKE_DIR),
            "--n",
            str(args.n),
        ]
    )

    rag_docs = read_jsonl(SMOKE_DIR / "rag_docs.jsonl")
    # Attach metadata for filtering: policies include policy_id; heuristics are shared.
    docs_for_ingest = []
    for d in rag_docs:
        md = {"id": d["id"], "type": d.get("type", "doc")}
        if "policy_id" in d:
            md["policy_id"] = d["policy_id"]
        docs_for_ingest.append({"id": d["id"], "text": d["text"], **md})

    # 2) Load config and init legs
    if not SMOKE_CONFIG.exists():
        raise FileNotFoundError(f"Missing config at {SMOKE_CONFIG}")
    raw_cfg = yaml.safe_load(SMOKE_CONFIG.read_text())
    cfg = TripodConfig(**raw_cfg)
    rag = RAGLeg(cfg.rag)
    prompter = PromptLeg(cfg.prompting)
    trainer = TrainingLeg(cfg.training)

    # 3) Ingest RAG docs
    rag.ingest(docs_for_ingest)

    # 4) Build a training file (prompt + target) with per-example retrieved context
    train_rows = read_jsonl(SMOKE_DIR / "train.jsonl")
    test_rows = read_jsonl(SMOKE_DIR / "test.jsonl")

    train_text_rows = []
    for row in train_rows:
        ctx = _build_rag_context(rag, row)
        prompt = prompter.run(
            {
                "domain": row["domain"],
                "rag_context": ctx,
                "sensor_data": {"policy_id": row["policy_id"], **row["sensor_data"]},
            }
        )
        target = json.dumps(row["expected"], ensure_ascii=False)
        train_text_rows.append({"text": f"{prompt}\n{target}"})

    train_text_path = SMOKE_DIR / "train_text.jsonl"
    write_jsonl(train_text_path, train_text_rows)

    # 5) Train adapter
    cfg.training.dataset_path = str(train_text_path)
    trainer.run()

    adapter_dir = Path(cfg.training.adapter_output_dir)

    # 6) Evaluate with and without RAG
    base_model, base_tok = load_llm(cfg.training.base_model, None)
    tuned_model, tuned_tok = load_llm(cfg.training.base_model, adapter_dir)

    base_with = evaluate(base_model, base_tok, prompter, rag, test_rows, use_rag=True, max_samples=args.eval_samples)
    base_without = evaluate(base_model, base_tok, prompter, rag, test_rows, use_rag=False, max_samples=args.eval_samples)
    tuned_with = evaluate(tuned_model, tuned_tok, prompter, rag, test_rows, use_rag=True, max_samples=args.eval_samples)
    tuned_without = evaluate(tuned_model, tuned_tok, prompter, rag, test_rows, use_rag=False, max_samples=args.eval_samples)

    logger.info("Base  +RAG: action=%.3f param=%.3f thermal_param=%.3f", base_with["action_acc"], base_with["param_acc"], base_with["thermal_param_acc"])
    logger.info("Base  -RAG: action=%.3f param=%.3f thermal_param=%.3f", base_without["action_acc"], base_without["param_acc"], base_without["thermal_param_acc"])
    logger.info("Tuned +RAG: action=%.3f param=%.3f thermal_param=%.3f", tuned_with["action_acc"], tuned_with["param_acc"], tuned_with["thermal_param_acc"])
    logger.info("Tuned -RAG: action=%.3f param=%.3f thermal_param=%.3f", tuned_without["action_acc"], tuned_without["param_acc"], tuned_without["thermal_param_acc"])


if __name__ == "__main__":
    main()
