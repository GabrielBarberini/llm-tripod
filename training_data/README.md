# training_data

Local, unversioned data and artifacts live here.

Suggested layout:
- `training_data/vectordb/`: local vector store files (docs + embeddings).
- `training_data/datasets/`: your domain train/test files (JSONL, parquet, etc).
- `training_data/adapters/`: optional local LoRA/QLoRA adapter outputs.
- `training_data/reports/`: evaluation artifacts and smoke-test reports.

Keep large datasets and adapters out of git; use this directory as a local
scratchpad referenced by your config paths.
