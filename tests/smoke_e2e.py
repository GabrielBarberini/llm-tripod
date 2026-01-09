"""
End-to-end smoke test: Training (LoRA/QLoRA) + RAFT + RAG + Prompting + Evaluation.
This script builds its own SFT dataset; it does not rely on prepare_train.

Designed to run on a GPU training node (e.g., RunPod).

Flow:
1) Generate synthetic dataset + RAG docs.
2) Ingest RAG docs into local vector store.
3) Build training files with and without RAFT retrieval (gated by raft.enabled).
4) Train LoRA adapters for no-RAFT and RAFT (if enabled).
5) Evaluate on the held-out set with two modes (rag.enabled on/off):
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
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.rag import RAGLeg
from core.training import TrainingLeg
from core.prompting import PromptLeg
from core.config import TripodConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("SmokeE2E")


DATASETS_DIR = ROOT / "training_data" / "datasets" / "smoke_e2e"
SMOKE_CONFIG = ROOT / "configs" / "smoke_e2e_config.yaml"
DEFAULT_REPORTS_DIR = ROOT / "training_data" / "reports" / "smoke_e2e"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--n",
        type=int,
        default=2000,
        help="Total synthetic samples to generate",
    )
    p.add_argument(
        "--eval-samples",
        type=int,
        default=200,
        help="Max evaluation samples (0 = all)",
    )
    p.add_argument(
        "--num-policies",
        type=int,
        default=50,
        help="Number of distinct policy docs",
    )
    p.add_argument(
        "--train-policy-ratio",
        type=float,
        default=0.8,
        help="Fraction of policies used for training split",
    )
    p.add_argument(
        "--holdout-policies",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled, test uses policy_ids not present in train (forces RAG).",
    )
    p.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of generated samples used for test split",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--report-dir",
        type=str,
        default="",
        help="Write report outputs here (default: training_data/reports/smoke_e2e/<timestamp>).",
    )
    p.add_argument(
        "--save-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write per-sample prediction JSONL files into the report dir.",
    )
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def build_query(example: dict[str, Any]) -> str:
    s = example["input_data"]
    return f"policy_id={example['policy_id']} temp={s['temp']} vib={s['vibration']} status={s['status']}"


def parse_action(generated: str) -> tuple[str, dict[str, Any]]:
    """Parse action and parameters from model output."""
    try:
        start = generated.find("{")
        end = generated.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(generated[start : end + 1])
            match obj:
                case {"action": action, "parameters": params}:
                    return str(action), params or {}
                case {"action": action}:
                    return str(action), {}
    except Exception:
        pass
    return "", {}


def _build_rag_context(rag: RAGLeg, row: dict[str, Any]) -> str:
    policy_ctx = rag.run(
        query=build_query(row), filters={"policy_id": row["policy_id"]}
    )
    heur_ctx = rag.run(query=build_query(row), filters={"type": "heuristic"})
    return "\n".join([c for c in [policy_ctx, heur_ctx] if c]).strip()


def _build_train_text_rows(
    prompter: PromptLeg,
    rag: RAGLeg,
    rows: list[dict[str, Any]],
    include_rag: bool,
) -> list[dict[str, str]]:
    train_text_rows = []
    for row in rows:
        task_label = row.get("task_label", "IoT")
        ctx = _build_rag_context(rag, row) if include_rag else ""
        prompt = prompter.render_prompt(
            {
                "task_label": task_label,
                "rag_context": ctx,
                "input_data": {
                    "policy_id": row["policy_id"],
                    **row["input_data"],
                },
            }
        )
        target = json.dumps(row["expected"], ensure_ascii=False)
        train_text_rows.append({"text": f"{prompt}\nASSISTANT:\n{target}"})
    return train_text_rows


def _ingest_rag_docs(rags: list[RAGLeg], docs: list[dict[str, Any]]) -> None:
    seen_paths: set[str] = set()
    for rag in rags:
        match rag.config.enabled:
            case False:
                continue
            case True:
                pass
        path = str(rag.config.vector_db_path)
        if path in seen_paths:
            continue
        rag.ingest(docs)
        seen_paths.add(path)


def _num(x: Any) -> float | None:
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
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )
    logging.getLogger().addHandler(handler)
    return log_path


def _safe_version(pkg: str) -> str | None:
    try:
        import importlib.metadata as importlib_metadata

        return importlib_metadata.version(pkg)
    except Exception:
        return None


def _git_info() -> dict[str, Any]:
    import subprocess

    info: dict[str, Any] = {}
    try:
        info["commit"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
            .decode()
            .strip()
        )
    except Exception:
        info["commit"] = None
    try:
        out = (
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)
            .decode()
            .strip()
        )
        info["dirty"] = bool(out)
    except Exception:
        info["dirty"] = None
    return info


def _env_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
    }

    for pkg in [
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "bitsandbytes",
        "sentence-transformers",
    ]:
        info[pkg] = _safe_version(pkg)

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


def _write_json(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    tmp.replace(path)


def _write_markdown(path: Path, report: dict[str, Any]):
    def _fmt(x: Any) -> str:
        if isinstance(x, float):
            return f"{x:.3f}"
        return str(x)

    metrics = report.get("metrics", {})
    lines: list[str] = []
    lines.append(f"# Tripod E2E Smoke Report ({report.get('run_id')})")
    lines.append("")
    lines.append(f"- Timestamp (UTC): `{report.get('timestamp_utc')}`")
    lines.append(
        f"- Git commit: `{report.get('git', {}).get('commit')}` (dirty={report.get('git', {}).get('dirty')})"
    )
    lines.append(
        f"- Args: `--n {report.get('args', {}).get('n')} --eval-samples {report.get('args', {}).get('eval_samples')}`"
    )
    lines.append(
        f"- RAFT enabled: `{report.get('retrieval', {}).get('raft_enabled')}`"
    )
    lines.append(
        f"- RAG enabled: `{report.get('retrieval', {}).get('rag_enabled')}`"
    )
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append(
        "| Pass | action_accuracy | param_accuracy | thermal_param_accuracy | samples | duration_s |"
    )
    lines.append(
        "|------|------------------|----------------|------------------------|---------|------------|"
    )

    def _row(pass_name: str, m: dict[str, Any]) -> str:
        return (
            f"| `{pass_name}` | {_fmt(m.get('action_accuracy'))} | {_fmt(m.get('param_accuracy'))} | {_fmt(m.get('thermal_param_accuracy'))} |"
            f" {_fmt(m.get('samples'))} | {_fmt(m.get('duration_s'))} |"
        )

    for key in [
        "base_with_rag",
        "base_without_rag",
        "tuned_no_raft_with_rag",
        "tuned_no_raft_without_rag",
        "tuned_raft_with_rag",
        "tuned_raft_without_rag",
    ]:
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


def _metric_delta(
    metrics: dict[str, dict[str, Any]],
    left: str,
    right: str,
    key: str,
) -> float | None:
    match (metrics.get(left), metrics.get(right)):
        case (dict() as left_map, dict() as right_map):
            left_val = left_map.get(key)
            right_val = right_map.get(key)
            if isinstance(left_val, (int, float)) and isinstance(
                right_val, (int, float)
            ):
                return float(left_val) - float(right_val)
    return None


def load_llm(model_id: str, adapter_dir: Path | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=(
            torch.float16 if torch.cuda.is_available() else torch.float32
        ),
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
    test_rows: list[dict[str, Any]],
    use_rag: bool,
    generation: dict[str, Any] | None = None,
    max_samples: int = 0,
    predictions_path: Path | None = None,
    pass_name: str = "",
) -> dict[str, float | int]:
    import torch

    gen_cfg = dict(generation or {})
    top_p = float(gen_cfg.get("top_p", 1.0))
    if "do_sample" in gen_cfg:
        do_sample = bool(gen_cfg["do_sample"])
    else:
        do_sample = top_p < 1.0

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
        task_label = row.get("task_label", "IoT")

        prompt = prompter.render_prompt(
            {
                "task_label": task_label,
                "rag_context": rag_context,
                "input_data": {
                    "policy_id": row["policy_id"],
                    **row["input_data"],
                },
            }
        )
        prompt = f"{prompt}\nASSISTANT:\n"

        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        out = model.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=do_sample,
            num_beams=1,
            repetition_penalty=1.1,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        action, params = parse_action(gen)

        expected = row["expected"]
        expected_action = expected["action"]
        expected_params = expected["parameters"]

        action_ok = action == expected_action
        if action_ok:
            action_hits += 1

        params_ok = False
        if action_ok and isinstance(params, dict):
            fan_pred = _num(params.get("fan_speed"))
            vb_pred = _num(params.get("voltage_bias"))
            fan_exp = _num(expected_params.get("fan_speed"))
            vb_exp = _num(expected_params.get("voltage_bias"))
            if (
                fan_pred is not None
                and vb_pred is not None
                and fan_exp is not None
                and vb_exp is not None
            ):
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
                        "task_label": row.get("task_label"),
                        "policy_id": row.get("policy_id"),
                        "input_data": row.get("input_data"),
                        "rag_context": rag_context,
                        "prompt": prompt,
                        "expected": expected,
                        "generated": gen,
                        "parsed": {"action": action, "parameters": params},
                        "match": {
                            "action_ok": action_ok,
                            "params_ok": params_ok,
                        },
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
        "action_accuracy": action_hits / n,
        "param_accuracy": param_hits / n,
        "thermal_param_accuracy": thermal_param_hits / max(1, thermal_total),
        "samples": len(test_rows),
        "duration_s": duration_s,
    }


def main():
    args = parse_args()
    run_id = _utc_timestamp()
    report_dir = (
        Path(args.report_dir).expanduser()
        if args.report_dir
        else (DEFAULT_REPORTS_DIR / run_id)
    )
    log_path = _setup_report_logging(report_dir)
    logger.info("Report directory: %s", report_dir)
    logger.info("Report log: %s", log_path)

    report: dict[str, Any] = {
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

    import subprocess

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            "python",
            str(ROOT / "tests" / "generate_smoke_dataset.py"),
            "--out-dir",
            str(DATASETS_DIR),
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
            (
                "--holdout-policies"
                if args.holdout_policies
                else "--no-holdout-policies"
            ),
        ]
    )

    rag_docs = read_jsonl(DATASETS_DIR / "rag_docs.jsonl")
    report["paths"].update(
        {
            "datasets_dir": str(DATASETS_DIR),
            "rag_docs": str(DATASETS_DIR / "rag_docs.jsonl"),
            "dataset_train": str(DATASETS_DIR / "train.jsonl"),
            "dataset_test": str(DATASETS_DIR / "test.jsonl"),
        }
    )
    _write_json(report_dir / "summary.json", report)

    docs_for_ingest = []
    for d in rag_docs:
        md = {"id": d["id"], "type": d.get("type", "doc")}
        if "policy_id" in d:
            md["policy_id"] = d["policy_id"]
        docs_for_ingest.append({"id": d["id"], "text": d["text"], **md})

    if not SMOKE_CONFIG.exists():
        raise FileNotFoundError(f"Missing config at {SMOKE_CONFIG}")
    raw_cfg = yaml.safe_load(SMOKE_CONFIG.read_text(encoding="utf-8"))
    match raw_cfg:
        case dict() as payload:
            cfg = TripodConfig.model_validate(payload)
        case _:
            raise ValueError("Smoke config root must be a mapping.")
    (report_dir / "config_snapshot.yaml").write_text(
        SMOKE_CONFIG.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    raft = RAGLeg(cfg.raft)
    rag = RAGLeg(cfg.rag)
    prompter = PromptLeg(cfg.prompting)
    trainer = TrainingLeg(cfg.training)
    report["paths"].update(
        {
            "raft_vectordb_path": str(cfg.raft.vector_db_path),
            "rag_vectordb_path": str(cfg.rag.vector_db_path),
            "base_model": str(cfg.training.base_model),
        }
    )
    report["retrieval"] = {
        "raft_enabled": bool(cfg.raft.enabled),
        "rag_enabled": bool(cfg.rag.enabled),
    }
    _write_json(report_dir / "summary.json", report)

    _ingest_rag_docs([raft, rag], docs_for_ingest)

    train_rows = read_jsonl(DATASETS_DIR / "train.jsonl")
    test_rows = read_jsonl(DATASETS_DIR / "test.jsonl")

    train_text_no_raft_path = DATASETS_DIR / "train_text_no_raft.jsonl"
    train_text_no_raft_rows = _build_train_text_rows(
        prompter, raft, train_rows, include_rag=False
    )
    write_jsonl(train_text_no_raft_path, train_text_no_raft_rows)
    report["paths"]["train_text_no_raft_jsonl"] = str(train_text_no_raft_path)

    train_text_raft_path: Path | None = None
    if cfg.raft.enabled:
        train_text_raft_path = DATASETS_DIR / "train_text_raft.jsonl"
        train_text_raft_rows = _build_train_text_rows(
            prompter, raft, train_rows, include_rag=True
        )
        write_jsonl(train_text_raft_path, train_text_raft_rows)
        report["paths"]["train_text_raft_jsonl"] = str(train_text_raft_path)

    _write_json(report_dir / "summary.json", report)

    adapter_base_dir = Path(cfg.training.adapter_output_dir)
    adapter_no_raft_dir = adapter_base_dir.with_name(
        f"{adapter_base_dir.name}_no_raft"
    )
    adapter_raft_dir = adapter_base_dir.with_name(
        f"{adapter_base_dir.name}_raft"
    )
    report["paths"]["adapter_output_dir_base"] = str(adapter_base_dir)
    report["paths"]["adapter_output_dir_no_raft"] = str(adapter_no_raft_dir)
    if cfg.raft.enabled:
        report["paths"]["adapter_output_dir_raft"] = str(adapter_raft_dir)
    _write_json(report_dir / "summary.json", report)

    def run_training(dataset_path: Path, adapter_dir: Path) -> float:
        cfg.training.dataset_path = str(dataset_path)
        cfg.training.adapter_output_dir = str(adapter_dir)
        t0 = time.time()
        trainer.run()
        return time.time() - t0

    training_report: dict[str, Any] = {}
    training_report["no_raft"] = {
        "dataset_path": str(train_text_no_raft_path),
        "adapter_dir": str(adapter_no_raft_dir),
        "duration_s": run_training(
            train_text_no_raft_path, adapter_no_raft_dir
        ),
    }
    if cfg.raft.enabled and train_text_raft_path is not None:
        training_report["raft"] = {
            "dataset_path": str(train_text_raft_path),
            "adapter_dir": str(adapter_raft_dir),
            "duration_s": run_training(train_text_raft_path, adapter_raft_dir),
        }

    report["training"] = training_report
    _write_json(report_dir / "summary.json", report)

    gen_settings = dict(cfg.evaluation.generation or {})
    report["generation"] = gen_settings
    _write_json(report_dir / "summary.json", report)

    preds_dir = report_dir / "predictions"

    def run_eval(
        model, tokenizer, use_rag: bool, pass_name: str
    ) -> dict[str, Any]:
        metrics = evaluate(
            model,
            tokenizer,
            prompter,
            rag,
            test_rows,
            use_rag=use_rag,
            generation=gen_settings,
            max_samples=args.eval_samples,
            predictions_path=(
                (preds_dir / f"{pass_name}.jsonl")
                if args.save_predictions
                else None
            ),
            pass_name=pass_name,
        )
        report["metrics"][pass_name] = metrics
        _write_json(report_dir / "summary.json", report)
        return metrics

    base_model, base_tok = load_llm(cfg.training.base_model, None)
    base_with = run_eval(base_model, base_tok, True, "base_with_rag")
    base_without = run_eval(base_model, base_tok, False, "base_without_rag")

    tuned_no_raft_model, tuned_no_raft_tok = load_llm(
        cfg.training.base_model, adapter_no_raft_dir
    )
    tuned_no_raft_with = run_eval(
        tuned_no_raft_model, tuned_no_raft_tok, True, "tuned_no_raft_with_rag"
    )
    tuned_no_raft_without = run_eval(
        tuned_no_raft_model,
        tuned_no_raft_tok,
        False,
        "tuned_no_raft_without_rag",
    )

    tuned_raft_with: dict[str, Any] | None = None
    tuned_raft_without: dict[str, Any] | None = None
    if cfg.raft.enabled:
        tuned_raft_model, tuned_raft_tok = load_llm(
            cfg.training.base_model, adapter_raft_dir
        )
        tuned_raft_with = run_eval(
            tuned_raft_model, tuned_raft_tok, True, "tuned_raft_with_rag"
        )
        tuned_raft_without = run_eval(
            tuned_raft_model,
            tuned_raft_tok,
            False,
            "tuned_raft_without_rag",
        )

    logger.info(
        "Base  +RAG: action=%.3f param=%.3f thermal_param=%.3f",
        base_with["action_accuracy"],
        base_with["param_accuracy"],
        base_with["thermal_param_accuracy"],
    )
    logger.info(
        "Base  -RAG: action=%.3f param=%.3f thermal_param=%.3f",
        base_without["action_accuracy"],
        base_without["param_accuracy"],
        base_without["thermal_param_accuracy"],
    )
    logger.info(
        "Tuned (no RAFT) +RAG: action=%.3f param=%.3f thermal_param=%.3f",
        tuned_no_raft_with["action_accuracy"],
        tuned_no_raft_with["param_accuracy"],
        tuned_no_raft_with["thermal_param_accuracy"],
    )
    logger.info(
        "Tuned (no RAFT) -RAG: action=%.3f param=%.3f thermal_param=%.3f",
        tuned_no_raft_without["action_accuracy"],
        tuned_no_raft_without["param_accuracy"],
        tuned_no_raft_without["thermal_param_accuracy"],
    )
    if tuned_raft_with is not None and tuned_raft_without is not None:
        logger.info(
            "Tuned (RAFT) +RAG: action=%.3f param=%.3f thermal_param=%.3f",
            tuned_raft_with["action_accuracy"],
            tuned_raft_with["param_accuracy"],
            tuned_raft_with["thermal_param_accuracy"],
        )
        logger.info(
            "Tuned (RAFT) -RAG: action=%.3f param=%.3f thermal_param=%.3f",
            tuned_raft_without["action_accuracy"],
            tuned_raft_without["param_accuracy"],
            tuned_raft_without["thermal_param_accuracy"],
        )

    metrics = report["metrics"]
    delta_defs = [
        (
            "rag_lift_base_param_accuracy",
            "base_with_rag",
            "base_without_rag",
            "param_accuracy",
        ),
        (
            "rag_lift_tuned_no_raft_param_accuracy",
            "tuned_no_raft_with_rag",
            "tuned_no_raft_without_rag",
            "param_accuracy",
        ),
        (
            "tune_gain_no_raft_with_rag_param_accuracy",
            "tuned_no_raft_with_rag",
            "base_with_rag",
            "param_accuracy",
        ),
        (
            "tune_gain_no_raft_without_rag_param_accuracy",
            "tuned_no_raft_without_rag",
            "base_without_rag",
            "param_accuracy",
        ),
    ]
    if cfg.raft.enabled:
        delta_defs.extend(
            [
                (
                    "rag_lift_tuned_raft_param_accuracy",
                    "tuned_raft_with_rag",
                    "tuned_raft_without_rag",
                    "param_accuracy",
                ),
                (
                    "tune_gain_raft_with_rag_param_accuracy",
                    "tuned_raft_with_rag",
                    "base_with_rag",
                    "param_accuracy",
                ),
                (
                    "tune_gain_raft_without_rag_param_accuracy",
                    "tuned_raft_without_rag",
                    "base_without_rag",
                    "param_accuracy",
                ),
                (
                    "raft_lift_with_rag_param_accuracy",
                    "tuned_raft_with_rag",
                    "tuned_no_raft_with_rag",
                    "param_accuracy",
                ),
                (
                    "raft_lift_without_rag_param_accuracy",
                    "tuned_raft_without_rag",
                    "tuned_no_raft_without_rag",
                    "param_accuracy",
                ),
            ]
        )

    deltas: dict[str, float] = {}
    for name, left, right, key in delta_defs:
        delta = _metric_delta(metrics, left, right, key)
        if delta is not None:
            deltas[name] = delta
    report["deltas"] = deltas
    _write_json(report_dir / "summary.json", report)
    _write_markdown(report_dir / "summary.md", report)
    logger.info("Report saved: %s", report_dir)


if __name__ == "__main__":
    main()
