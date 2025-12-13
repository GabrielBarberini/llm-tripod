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
import platform
import time
from datetime import datetime, timezone
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
DEFAULT_REPORTS_DIR = SMOKE_DIR / "reports"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000, help="Total synthetic samples to generate")
    p.add_argument("--eval-samples", type=int, default=200, help="Max evaluation samples (0 = all)")
    p.add_argument("--num-policies", type=int, default=50, help="Number of distinct policy docs")
    p.add_argument("--train-policy-ratio", type=float, default=0.8, help="Fraction of policies used for training split")
    p.add_argument(
        "--holdout-policies",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled, test uses policy_ids not present in train (forces RAG).",
    )
    p.add_argument("--test-ratio", type=float, default=0.2, help="Fraction of generated samples used for test split")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--report-dir",
        type=str,
        default="",
        help="Write report outputs here (default: training_data/smoke/reports/<timestamp>).",
    )
    p.add_argument(
        "--save-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write per-sample prediction JSONL files into the report dir.",
    )
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


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _setup_report_logging(report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    log_path = report_dir / "run.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(handler)
    return log_path


def _safe_version(pkg: str) -> Optional[str]:
    try:
        import importlib.metadata as importlib_metadata

        return importlib_metadata.version(pkg)
    except Exception:
        return None


def _git_info() -> Dict[str, Any]:
    import subprocess

    info: Dict[str, Any] = {}
    try:
        info["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        info["commit"] = None
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).decode().strip()
        info["dirty"] = bool(out)
    except Exception:
        info["dirty"] = None
    return info


def _env_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
    }

    # Package versions (best-effort)
    for pkg in ["torch", "transformers", "datasets", "accelerate", "peft", "bitsandbytes", "sentence-transformers"]:
        info[pkg] = _safe_version(pkg)

    # CUDA/GPU details (best-effort)
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = getattr(torch.version, "cuda", None)
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
    except Exception:
        pass

    return info


def _write_json(path: Path, payload: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _write_markdown(path: Path, report: Dict[str, Any]):
    def _fmt(x: Any) -> str:
        if isinstance(x, float):
            return f"{x:.3f}"
        return str(x)

    metrics = report.get("metrics", {})
    lines: List[str] = []
    lines.append(f"# Tripod E2E Smoke Report ({report.get('run_id')})")
    lines.append("")
    lines.append(f"- Timestamp (UTC): `{report.get('timestamp_utc')}`")
    lines.append(f"- Git commit: `{report.get('git', {}).get('commit')}` (dirty={report.get('git', {}).get('dirty')})")
    lines.append(f"- Args: `--n {report.get('args', {}).get('n')} --eval-samples {report.get('args', {}).get('eval_samples')}`")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Pass | action_acc | param_acc | thermal_param_acc | samples | duration_s |")
    lines.append("|------|------------|-----------|------------------|---------|------------|")

    def _row(pass_name: str, m: Dict[str, Any]) -> str:
        return (
            f"| `{pass_name}` | {_fmt(m.get('action_acc'))} | {_fmt(m.get('param_acc'))} | {_fmt(m.get('thermal_param_acc'))} |"
            f" {_fmt(m.get('samples'))} | {_fmt(m.get('duration_s'))} |"
        )

    for key in ["base_with_rag", "base_without_rag", "tuned_with_rag", "tuned_without_rag"]:
        if key in metrics:
            lines.append(_row(key, metrics[key]))

    if "deltas" in report:
        lines.append("")
        lines.append("## Deltas")
        lines.append("")
        for k, v in report["deltas"].items():
            lines.append(f"- `{k}`: {_fmt(v)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


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
    predictions_path: Optional[Path] = None,
    pass_name: str = "",
) -> Dict[str, float]:
    import torch

    if max_samples and max_samples > 0:
        test_rows = test_rows[:max_samples]

    action_hits = 0
    param_hits = 0
    thermal_param_hits = 0
    thermal_total = 0

    preds_f = None
    if predictions_path is not None:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        preds_f = predictions_path.open("w", encoding="utf-8", buffering=1)

    start_s = time.time()
    for row in test_rows:
        rag_context = _build_rag_context(rag, row) if use_rag else ""

        prompt = prompter.run(
            {
                "domain": row["domain"],
                "rag_context": rag_context,
                "sensor_data": {"policy_id": row["policy_id"], **row["sensor_data"]},
            }
        )
        prompt = f"{prompt}\nASSISTANT:\n"

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

        if preds_f is not None:
            preds_f.write(
                json.dumps(
                    {
                        "pass": pass_name,
                        "use_rag": use_rag,
                        "domain": row.get("domain"),
                        "policy_id": row.get("policy_id"),
                        "sensor_data": row.get("sensor_data"),
                        "rag_context": rag_context,
                        "prompt": prompt,
                        "expected": expected,
                        "generated": gen,
                        "parsed": {"action": action, "parameters": params},
                        "match": {"action_ok": action_ok, "params_ok": params_ok},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    if preds_f is not None:
        preds_f.close()

    duration_s = time.time() - start_s
    n = max(1, len(test_rows))
    return {
        "action_acc": action_hits / n,
        "param_acc": param_hits / n,
        "thermal_param_acc": thermal_param_hits / max(1, thermal_total),
        "samples": len(test_rows),
        "duration_s": duration_s,
    }


def main():
    args = parse_args()
    run_id = _utc_timestamp()
    report_dir = Path(args.report_dir).expanduser() if args.report_dir else (DEFAULT_REPORTS_DIR / run_id)
    log_path = _setup_report_logging(report_dir)
    logger.info("Report directory: %s", report_dir)
    logger.info("Report log: %s", log_path)

    report: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "args": {
            "n": args.n,
            "eval_samples": args.eval_samples,
            "num_policies": args.num_policies,
            "train_policy_ratio": args.train_policy_ratio,
            "holdout_policies": bool(args.holdout_policies),
            "test_ratio": args.test_ratio,
            "seed": args.seed,
            "report_dir": str(report_dir),
            "save_predictions": bool(args.save_predictions),
        },
        "git": _git_info(),
        "env": _env_info(),
        "paths": {},
        "metrics": {},
    }
    _write_json(report_dir / "summary.json", report)

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
            "--num-policies",
            str(args.num_policies),
            "--train-policy-ratio",
            str(args.train_policy_ratio),
            "--test-ratio",
            str(args.test_ratio),
            "--seed",
            str(args.seed),
            "--holdout-policies" if args.holdout_policies else "--no-holdout-policies",
        ]
    )

    rag_docs = read_jsonl(SMOKE_DIR / "rag_docs.jsonl")
    report["paths"].update(
        {
            "smoke_dir": str(SMOKE_DIR),
            "rag_docs": str(SMOKE_DIR / "rag_docs.jsonl"),
            "dataset_train": str(SMOKE_DIR / "train.jsonl"),
            "dataset_test": str(SMOKE_DIR / "test.jsonl"),
        }
    )
    _write_json(report_dir / "summary.json", report)

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
    (report_dir / "config_snapshot.yaml").write_text(SMOKE_CONFIG.read_text(), encoding="utf-8")
    rag = RAGLeg(cfg.rag)
    prompter = PromptLeg(cfg.prompting)
    trainer = TrainingLeg(cfg.training)
    report["paths"].update(
        {
            "vectordb_path": str(cfg.rag.vector_db_path),
            "adapter_output_dir": str(cfg.training.adapter_output_dir),
            "base_model": str(cfg.training.base_model),
        }
    )
    _write_json(report_dir / "summary.json", report)

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
        train_text_rows.append({"text": f"{prompt}\nASSISTANT:\n{target}"})

    train_text_path = SMOKE_DIR / "train_text.jsonl"
    write_jsonl(train_text_path, train_text_rows)
    report["paths"]["train_text_jsonl"] = str(train_text_path)
    _write_json(report_dir / "summary.json", report)

    # 5) Train adapter
    cfg.training.dataset_path = str(train_text_path)
    t0 = time.time()
    trainer.run()
    report["training"] = {"duration_s": time.time() - t0}
    _write_json(report_dir / "summary.json", report)

    adapter_dir = Path(cfg.training.adapter_output_dir)

    # 6) Evaluate with and without RAG
    base_model, base_tok = load_llm(cfg.training.base_model, None)
    tuned_model, tuned_tok = load_llm(cfg.training.base_model, adapter_dir)

    preds_dir = report_dir / "predictions"

    base_with = evaluate(
        base_model,
        base_tok,
        prompter,
        rag,
        test_rows,
        use_rag=True,
        max_samples=args.eval_samples,
        predictions_path=(preds_dir / "base_with_rag.jsonl") if args.save_predictions else None,
        pass_name="base_with_rag",
    )
    report["metrics"]["base_with_rag"] = base_with
    _write_json(report_dir / "summary.json", report)

    base_without = evaluate(
        base_model,
        base_tok,
        prompter,
        rag,
        test_rows,
        use_rag=False,
        max_samples=args.eval_samples,
        predictions_path=(preds_dir / "base_without_rag.jsonl") if args.save_predictions else None,
        pass_name="base_without_rag",
    )
    report["metrics"]["base_without_rag"] = base_without
    _write_json(report_dir / "summary.json", report)

    tuned_with = evaluate(
        tuned_model,
        tuned_tok,
        prompter,
        rag,
        test_rows,
        use_rag=True,
        max_samples=args.eval_samples,
        predictions_path=(preds_dir / "tuned_with_rag.jsonl") if args.save_predictions else None,
        pass_name="tuned_with_rag",
    )
    report["metrics"]["tuned_with_rag"] = tuned_with
    _write_json(report_dir / "summary.json", report)

    tuned_without = evaluate(
        tuned_model,
        tuned_tok,
        prompter,
        rag,
        test_rows,
        use_rag=False,
        max_samples=args.eval_samples,
        predictions_path=(preds_dir / "tuned_without_rag.jsonl") if args.save_predictions else None,
        pass_name="tuned_without_rag",
    )
    report["metrics"]["tuned_without_rag"] = tuned_without
    _write_json(report_dir / "summary.json", report)

    logger.info("Base  +RAG: action=%.3f param=%.3f thermal_param=%.3f", base_with["action_acc"], base_with["param_acc"], base_with["thermal_param_acc"])
    logger.info("Base  -RAG: action=%.3f param=%.3f thermal_param=%.3f", base_without["action_acc"], base_without["param_acc"], base_without["thermal_param_acc"])
    logger.info("Tuned +RAG: action=%.3f param=%.3f thermal_param=%.3f", tuned_with["action_acc"], tuned_with["param_acc"], tuned_with["thermal_param_acc"])
    logger.info("Tuned -RAG: action=%.3f param=%.3f thermal_param=%.3f", tuned_without["action_acc"], tuned_without["param_acc"], tuned_without["thermal_param_acc"])

    report["deltas"] = {
        "rag_lift_base_param_acc": base_with["param_acc"] - base_without["param_acc"],
        "rag_lift_tuned_param_acc": tuned_with["param_acc"] - tuned_without["param_acc"],
        "tune_gain_with_rag_param_acc": tuned_with["param_acc"] - base_with["param_acc"],
        "tune_gain_without_rag_param_acc": tuned_without["param_acc"] - base_without["param_acc"],
    }
    _write_json(report_dir / "summary.json", report)
    _write_markdown(report_dir / "summary.md", report)
    logger.info("Report saved: %s", report_dir)


if __name__ == "__main__":
    main()
