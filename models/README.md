# Models Directory

This directory stores trained 3D CNN model weights, architecture artifacts, and training checkpoints.

## Directory Structure

```
models/
└── checkpoints/   # ModelCheckpoint saves (e.g. best_model.keras)
```

## Checkpoint Generation

Training checkpoints are automatically populated by `ModelTrainer` in `src/model/trainer.py` when running:

```bash
python main.py train
```
