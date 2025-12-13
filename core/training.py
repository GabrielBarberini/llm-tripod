import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from core.base import BaseLeg
from core.config import TrainingConfig

logger = logging.getLogger(__name__)


class TrainingLeg(BaseLeg):
    """
    Leg 1: Handles model fine-tuning (LoRA/PEFT).
    Integrate your trainer of choice inside `run`.
    """

    def __init__(self, config: TrainingConfig):
        super().__init__(config)

    def run(self, input_data: Any = None):
        if not self.config.enabled:
            logger.info("Training leg disabled. Skipping.")
            return

        # Lazy imports: these are heavy and only needed on training nodes.
        from datasets import load_dataset
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
            default_data_collator,
        )

        logger.info("Starting training on base model: %s", self.config.base_model)
        logger.info("Dataset path: %s", self.config.dataset_path)
        logger.info("Adapter output dir: %s", self.config.adapter_output_dir)

        hp = dict(self.config.hyperparameters or {})
        output_dir = Path(self.config.adapter_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        use_cuda = torch.cuda.is_available()
        quant = str(hp.get("quantization", "4bit")).lower()
        use_4bit = use_cuda and quant == "4bit"

        bnb_config: Optional[BitsAndBytesConfig] = None
        model_load_kwargs: Dict[str, Any] = {"low_cpu_mem_usage": True}
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model_load_kwargs.update({"quantization_config": bnb_config, "device_map": "auto"})
            logger.info("Using QLoRA (4-bit NF4) on CUDA.")
        else:
            logger.info("Using full precision / non-4bit training (cuda=%s, quant=%s).", use_cuda, quant)

        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.float16 if use_cuda else torch.float32,
            **model_load_kwargs,
        )
        model.config.pad_token_id = tokenizer.eos_token_id

        if use_4bit:
            model = prepare_model_for_kbit_training(model)

        lora_cfg = LoraConfig(
            r=int(self.config.lora_config.r),
            lora_alpha=int(self.config.lora_config.alpha),
            lora_dropout=float(self.config.lora_config.dropout),
            target_modules=list(self.config.lora_config.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)

        dataset_path = self.config.dataset_path
        ext = Path(dataset_path).suffix.lower()
        ds_format = "json" if ext in {".json", ".jsonl"} else "text"
        ds = load_dataset(ds_format, data_files={"train": dataset_path})["train"]

        def to_text(row):
            # Accept either {"text": "..."} or our smoke format:
            # {"domain","device_profile","sensor_data","rag_context","expected":{...}}
            if "text" in row:
                return {"text": row["text"]}
            expected = row.get("expected", {})
            prompt = {
                "domain": row.get("domain", "Thermal Control"),
                "device_profile": row.get("device_profile", "eco"),
                "sensor_data": row.get("sensor_data", {}),
                "rag_context": row.get("rag_context", ""),
            }
            system = (
                "You are a safety-first Industrial IoT controller.\n"
                "Return ONLY valid JSON with keys: action, parameters, reasoning."
            )
            user = (
                f"DEVICE_PROFILE: {prompt['device_profile']}\n"
                f"HISTORY:\n{prompt['rag_context']}\n"
                f"SENSOR:\n{json.dumps(prompt['sensor_data'])}\n"
                "OUTPUT JSON:"
            )
            target = json.dumps(
                {"action": expected.get("action"), "parameters": expected.get("parameters"), "reasoning": expected.get("reasoning")},
                ensure_ascii=False,
            )
            return {"text": f"SYSTEM:\n{system}\n\nUSER:\n{user}\n{target}"}

        ds = ds.map(to_text, remove_columns=ds.column_names)

        max_len = int(hp.get("max_seq_length", 256))
        response_marker = str(hp.get("response_marker", "\nASSISTANT:\n"))
        mask_prompt = bool(hp.get("mask_prompt", True))

        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        def _encode_one(text: str):
            # Optional prompt/completion split for SFT-style training.
            marker_idx = text.find(response_marker) if response_marker else -1
            has_marker = marker_idx >= 0
            do_mask = bool(mask_prompt and has_marker)
            if has_marker:
                prompt_part = text[: marker_idx + len(response_marker)]
                completion_part = text[marker_idx + len(response_marker) :]
            else:
                prompt_part = ""
                completion_part = text

            prompt_ids = tokenizer(prompt_part, add_special_tokens=False).input_ids if prompt_part else []
            completion_ids = tokenizer(completion_part, add_special_tokens=False).input_ids if completion_part else []

            bos = [bos_id] if bos_id is not None else []
            eos = [eos_id] if eos_id is not None else []

            max_body = max_len - len(bos) - len(eos)
            if max_body <= 0:
                raise ValueError(f"max_seq_length too small (need > {len(bos) + len(eos)}).")

            if len(completion_ids) > max_body:
                # Keep the tail so the model still learns the end of the completion.
                completion_ids = completion_ids[-max_body:]
                prompt_ids = []
            else:
                remaining_prompt = max_body - len(completion_ids)
                if len(prompt_ids) > remaining_prompt:
                    prompt_ids = prompt_ids[-remaining_prompt:]

            input_ids = bos + prompt_ids + completion_ids + eos
            attention_mask = [1] * len(input_ids)

            if do_mask:
                labels = ([-100] * (len(bos) + len(prompt_ids))) + completion_ids + eos
            else:
                labels = input_ids.copy()

            # Pad to max_len and ensure padding tokens do not contribute to loss.
            pad_n = max_len - len(input_ids)
            if pad_n > 0:
                input_ids = input_ids + ([pad_id] * pad_n)
                attention_mask = attention_mask + ([0] * pad_n)
                labels = labels + ([-100] * pad_n)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        def tokenize(batch):
            input_ids = []
            attention_mask = []
            labels = []
            for text in batch["text"]:
                enc = _encode_one(text)
                input_ids.append(enc["input_ids"])
                attention_mask.append(enc["attention_mask"])
                labels.append(enc["labels"])
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        ds = ds.map(tokenize, batched=True, remove_columns=["text"])

        data_collator = default_data_collator

        args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=int(hp.get("micro_batch_size", 1)),
            gradient_accumulation_steps=int(hp.get("gradient_accumulation_steps", 4)),
            num_train_epochs=float(hp.get("num_epochs", 1)),
            learning_rate=float(hp.get("learning_rate", 2e-4)),
            logging_steps=int(hp.get("logging_steps", 10)),
            save_strategy="no",
            report_to="none",
            bf16=bool(hp.get("bf16", True)) and use_cuda,
            fp16=bool(hp.get("fp16", False)) and use_cuda,
            gradient_checkpointing=True,
            optim=(
                "adamw_torch"
                if not use_cuda
                else str(hp.get("optim") or ("paged_adamw_8bit" if use_4bit else "adamw_torch"))
            ),
            lr_scheduler_type=str(hp.get("lr_scheduler_type", "cosine")),
            warmup_ratio=float(hp.get("warmup_ratio", 0.03)),
        )

        trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=data_collator)
        trainer.train()
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        logger.info("Training complete. Adapter saved to %s", output_dir)
