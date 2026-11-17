import os
import json
from datetime import datetime, timezone
from typing import Dict, Any
import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from app.core.config import settings
from app.core.logging import logger

router = APIRouter()

# Global training execution state tracker
training_state = {
    "is_training": False,
    "current_epoch": 0,
    "total_epochs": 5,
    "current_step": 0,
    "total_steps": None,
    "dataset_name": None,
    "status": "IDLE",
    "last_accuracy": None,
    "last_loss": None,
    "checkpoint_path": None,
    "message": "Ready to train or evaluate 3D CNN model.",
}
_dataset_cache = {"key": None, "value": None}


def read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def write_json_atomically(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, path)


def get_dataset_info(root_dir: str, dataset_name):
    if not dataset_name:
        return {"available": False, "size_mb": None, "splits": None, "volume_shape": None}

    dataset_path = os.path.join(root_dir, "data", f"{dataset_name}.npz")
    if not os.path.isfile(dataset_path):
        _dataset_cache.update(key=None, value=None)
        return {"available": False, "size_mb": None, "splits": None, "volume_shape": None}

    stat = os.stat(dataset_path)
    cache_key = (dataset_path, stat.st_mtime_ns, stat.st_size)
    if _dataset_cache["key"] != cache_key:
        split_counts = {}
        volume_shape = None
        try:
            with np.load(dataset_path, allow_pickle=False) as dataset:
                for split in ("train", "val", "test"):
                    image_key = f"{split}_images"
                    if image_key in dataset.files:
                        shape = dataset[image_key].shape
                        split_counts[split] = int(shape[0])
                        volume_shape = list(shape[1:])
        except (OSError, ValueError, KeyError):
            split_counts = None
            volume_shape = None
        _dataset_cache.update(key=cache_key, value={"splits": split_counts, "volume_shape": volume_shape})

    cached = _dataset_cache["value"] or {}
    return {
        "available": True,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "splits": cached.get("splits"),
        "volume_shape": cached.get("volume_shape"),
    }


def append_training_run(root_dir: str, run: Dict[str, Any]) -> None:
    history_path = os.path.join(root_dir, "results", "training_runs.json")
    history = read_json(history_path)
    if not isinstance(history, list):
        history = []
    history.append(run)
    write_json_atomically(history_path, history)


def get_training_runs(root_dir: str, dataset_name):
    runs_path = os.path.join(root_dir, "results", "training_runs.json")
    runs = read_json(runs_path)
    if isinstance(runs, list):
        return runs

    # Preserve the repository's existing recorded Keras history as a legacy run.
    history_path = os.path.join(root_dir, "results", "training_history.json")
    metrics = read_json(history_path)
    if not isinstance(metrics, dict) or not metrics.get("accuracy"):
        return []
    recorded_at = datetime.fromtimestamp(os.path.getmtime(history_path), timezone.utc).isoformat()
    legacy = [{
        "run_id": "existing-training-history",
        "recorded_at": recorded_at,
        "status": "RECORDED",
        "dataset_name": dataset_name,
        "requested_epochs": len(metrics["accuracy"]),
        "completed_epochs": len(metrics["accuracy"]),
        "metrics": {
            key: metrics[key]
            for key in ("accuracy", "loss", "val_accuracy", "val_loss")
            if isinstance(metrics.get(key), list)
        },
    }]
    write_json_atomically(runs_path, legacy)
    return legacy

def get_root_dir() -> str:
    root = getattr(settings, "PROJECT_ROOT", None)
    if root and os.path.exists(os.path.join(root, "main.py")):
        return root
    return settings.BASE_DIR

def get_latest_checkpoint_info() -> Dict[str, Any]:
    root_dir = get_root_dir()
    ckpt_path = os.path.join(root_dir, "models", "checkpoints", "best_model.keras")
    exists = os.path.exists(ckpt_path)
    size_mb = round(os.path.getsize(ckpt_path) / (1024 * 1024), 2) if exists else 0
    mtime = os.path.getmtime(ckpt_path) if exists else None

    history_path = os.path.join(root_dir, "results", "training_history.json")
    history_data = read_json(history_path)

    return {
        "checkpoint_exists": exists,
        "checkpoint_path": ckpt_path if exists else None,
        "size_mb": size_mb,
        "last_modified": mtime,
        "history": history_data
    }

@router.get("/status", summary="Get 3D CNN Model Training & Checkpoint Status")
def get_training_status():
    root_dir = get_root_dir()
    ckpt_info = get_latest_checkpoint_info()

    config_path = os.path.join(root_dir, "config", "config.yaml")
    config = read_config(config_path)
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    dataset_name = data_config.get("dataset_name")
    class_labels = data_config.get("class_labels")
    dataset_info = get_dataset_info(root_dir, dataset_name)
    training_state["dataset_name"] = dataset_name

    evaluation_path = os.path.join(root_dir, "results", "evaluation_metrics_test.json")
    evaluation_metrics = read_json(evaluation_path)
    checkpoint_path = ckpt_info.get("checkpoint_path")
    if evaluation_metrics and checkpoint_path and os.path.exists(evaluation_path):
        if os.path.getmtime(evaluation_path) < os.path.getmtime(checkpoint_path):
            evaluation_metrics = None

    run_history = get_training_runs(root_dir, dataset_name)

    architecture = {
        "input_shape": model_config.get("input_shape"),
        "filters": model_config.get("filters"),
        "kernel_size": model_config.get("kernel_size"),
        "dense_units": model_config.get("dense_units"),
        "dropout_rate": model_config.get("dropout_rate"),
        "l2_reg": model_config.get("l2_reg"),
        "batch_size": training_config.get("batch_size"),
        "optimizer": training_config.get("optimizer"),
        "loss": training_config.get("loss"),
    }

    return {
        **training_state,
        "dataset_name": dataset_name,
        "dataset_available": dataset_info["available"],
        "dataset_size_mb": dataset_info["size_mb"],
        "dataset_splits": dataset_info["splits"],
        "volume_shape": dataset_info["volume_shape"],
        "dataset_classes": class_labels,
        "dataset_source": data_config.get("source"),
        "target_size": config.get("preprocessing", {}).get("target_size"),
        "architecture": architecture,
        "evaluation_metrics": evaluation_metrics,
        "run_history": run_history,
        "checkpoint": ckpt_info,
    }


def read_config(config_path: str) -> Dict[str, Any]:
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config if isinstance(config, dict) else {}
    except (OSError, ImportError, ValueError):
        return {}

def run_background_train_task(epochs: int = 5):
    global training_state
    root_dir = get_root_dir()
    started_at = datetime.now(timezone.utc)
    history_path = os.path.join(root_dir, "results", "training_history.json")
    previous_history_mtime = os.path.getmtime(history_path) if os.path.exists(history_path) else None
    try:
        training_state["is_training"] = True
        training_state["status"] = "TRAINING"
        training_state["total_epochs"] = epochs
        training_state["message"] = f"Training 3D CNN on {training_state.get('dataset_name') or 'configured dataset'} for up to {epochs} epochs..."
        
        # Execute training pipeline
        import subprocess
        python_exe = os.path.join(root_dir, "venv", "Scripts", "python.exe")
        if not os.path.exists(python_exe):
            python_exe = "python"
            
        cmd = [python_exe, "main.py", "train", "--epochs", str(epochs)]
        result = subprocess.run(cmd, cwd=root_dir, capture_output=True, text=True)
        
        if result.returncode == 0:
            training_state["status"] = "COMPLETED"
            training_state["message"] = "Model trained successfully. Checkpoint saved."
            latest_history = read_json(history_path)
            latest_mtime = os.path.getmtime(history_path) if os.path.exists(history_path) else None
            if latest_history is None or latest_mtime == previous_history_mtime:
                raise RuntimeError("Training completed but did not produce a new training history file.")
        else:
            training_state["status"] = "ERROR"
            training_state["message"] = f"Training exited with code {result.returncode}: {result.stderr[:200]}"
    except Exception as e:
        training_state["status"] = "ERROR"
        training_state["message"] = str(e)
    finally:
        training_state["is_training"] = False
        finished_at = datetime.now(timezone.utc)
        actual_history = read_json(history_path)
        latest_mtime = os.path.getmtime(history_path) if os.path.exists(history_path) else None
        if latest_mtime == previous_history_mtime:
            actual_history = None
        accuracy = (actual_history or {}).get("accuracy", [])
        loss = (actual_history or {}).get("loss", [])
        val_accuracy = (actual_history or {}).get("val_accuracy", [])
        val_loss = (actual_history or {}).get("val_loss", [])
        training_state["last_accuracy"] = accuracy[-1] if accuracy else None
        training_state["last_loss"] = loss[-1] if loss else None
        try:
            append_training_run(root_dir, {
                "run_id": f"{started_at.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}",
                "started_at": started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
                "status": training_state["status"],
                "dataset_name": training_state.get("dataset_name"),
                "requested_epochs": epochs,
                "completed_epochs": len(accuracy),
                "metrics": {
                    "accuracy": accuracy,
                    "loss": loss,
                    "val_accuracy": val_accuracy,
                    "val_loss": val_loss,
                } if actual_history else None,
                "error": training_state["message"] if training_state["status"] == "ERROR" else None,
            })
        except OSError as exc:
            logger.exception("Could not persist training run history: %s", exc)

@router.post("/start", summary="Trigger Background 3D CNN Model Training")
def start_model_training(background_tasks: BackgroundTasks, epochs: int = Query(default=5, ge=1, le=100)):
    global training_state
    if training_state["is_training"]:
        raise HTTPException(status_code=409, detail="Model training is already in progress.")

    background_tasks.add_task(run_background_train_task, epochs)
    training_state["is_training"] = True
    training_state["status"] = "STARTING"
    training_state["message"] = "Starting model training in background..."
    return {"status": "STARTED", "epochs": epochs, "message": "Model training initiated."}
