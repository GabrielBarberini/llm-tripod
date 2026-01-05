from __future__ import annotations

from typing import Any

from pydantic import BaseModel, model_validator


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
    few_shot: dict[str, Any]
    backend: str = "raw"
    dspy: DSPyConfig | None = None

    class Config:
        extra = "ignore"


class EvaluationConfig(BaseModel):
    test_set_path: str
    metrics: list[str]
    generation: dict[str, Any] | None = None

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

    @model_validator(mode="before")
    @classmethod
    def _coerce_retrieval_configs(cls, data: Any) -> Any:
        match data:
            case {
                "rag": {
                    "training": dict() as training,
                    "inference": dict() as inference,
                },
                **rest,
            }:
                return {**rest, "rag": inference, "raft": training}
            case {"rag": {"training": dict() as training} as rag, **rest} if (
                "inference" not in rag
            ):
                return {
                    **rest,
                    "rag": training,
                    "raft": training,
                }
            case {
                "rag": {"inference": dict() as inference} as rag,
                **rest,
            } if ("training" not in rag):
                return {
                    **rest,
                    "rag": inference,
                    "raft": inference,
                }
            case {"rag": dict() as rag, "raft": dict() as raft, **rest}:
                return {**rest, "rag": rag, "raft": raft}
            case {"rag": dict() as rag, **rest} if "raft" not in data:
                return {**rest, "rag": rag, "raft": rag}
            case {"raft": dict() as raft, **rest} if "rag" not in data:
                return {**rest, "rag": raft, "raft": raft}
            case _:
                return data
