from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class LoRAConfigModel(BaseModel):
    r: int
    alpha: int
    dropout: float
    target_modules: list[str]
    bias: str | None = None
    task_type: str | None = None

    class Config:
        extra = "ignore"


class TrainingConfig(BaseModel):
    enabled: bool
    base_model: str
    dataset_path: str
    adapter_output_dir: str
    lora_config: LoRAConfigModel
    hyperparameters: dict[str, Any]

    class Config:
        extra = "ignore"


class RAGConfig(BaseModel):
    enabled: bool
    vector_db_type: str
    vector_db_path: str
    collection_name: str
    ingestion: dict[str, Any]
    retrieval: dict[str, Any]

    class Config:
        extra = "ignore"


class DSPyConfig(BaseModel):
    instructions: str | None = None
    include_user_prompt: bool = True
    chain_of_thought: bool = False
    output_field: str = "response"
    output_desc: str = "Model response."

    class Config:
        extra = "ignore"


class PromptingConfig(BaseModel):
    template_id: str
    system_prompt: str
    user_prompt_structure: str
    backend: str = "raw"
    dspy: DSPyConfig | None = None

    class Config:
        extra = "ignore"


class EvaluationConfig(BaseModel):
    test_set_path: str
    metrics: list[str]
    generation: dict[str, Any] | None = None
    entrypoint: str | None = None
    evaluator: str | None = None

    class Config:
        extra = "ignore"


class TripodConfig(BaseModel):
    experiment_name: str
    output_dir: str
    training: TrainingConfig
    rag: RAGConfig
    raft: RAGConfig
    prompting: PromptingConfig
    evaluation: EvaluationConfig

    class Config:
        extra = "ignore"
