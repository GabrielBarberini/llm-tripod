from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class LoRAConfigModel(BaseModel):
    r: int
    alpha: int
    dropout: float
    target_modules: List[str]
    bias: Optional[str] = None
    task_type: Optional[str] = None

    class Config:
        extra = "ignore"


class TrainingConfig(BaseModel):
    enabled: bool
    base_model: str
    dataset_path: str
    adapter_output_dir: str
    lora_config: LoRAConfigModel
    hyperparameters: Dict[str, Any]

    class Config:
        extra = "ignore"


class RAGConfig(BaseModel):
    enabled: bool
    vector_db_type: str
    vector_db_path: str
    collection_name: str
    ingestion: Dict[str, Any]
    retrieval: Dict[str, Any]

    class Config:
        extra = "ignore"


class PromptingConfig(BaseModel):
    template_id: str
    system_prompt: str
    user_prompt_structure: str
    few_shot: Dict[str, Any]

    class Config:
        extra = "ignore"


class EvaluationConfig(BaseModel):
    test_set_path: str
    metrics: List[str]

    class Config:
        extra = "ignore"


class TripodConfig(BaseModel):
    experiment_name: str
    output_dir: str
    training: TrainingConfig
    rag: RAGConfig
    prompting: PromptingConfig
    evaluation: EvaluationConfig

    class Config:
        extra = "ignore"
