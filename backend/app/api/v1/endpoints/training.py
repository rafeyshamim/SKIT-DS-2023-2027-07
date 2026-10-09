import os
import json
import glob
from typing import Dict, Any, Optional
from fastapi import APIRouter, BackgroundTasks
from app.core.config import settings
from app.core.logging import logger

router = APIRouter()

# Global training execution state tracker
training_state = {
    "is_training": False,
    "current_epoch": 0,
    "total_epochs": 5,
    "current_step": 0,
    "total_steps": 61,
    "dataset_name": "organmnist3d",
    "status": "IDLE",
    "last_accuracy": None,
    "last_loss": None,
    "checkpoint_path": None,
    "message": "Ready to train or evaluate 3D CNN model.",
}

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
    history_data = None
    if os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history_data = json.load(f)
        except Exception:
            pass

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
    dataset_name = "nodulemnist3d"
    class_labels = ["Benign Pulmonary Nodule", "Malignant Lung Cancer"]
    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                dataset_name = cfg.get("data", {}).get("dataset_name", "nodulemnist3d")
                class_labels = cfg.get("data", {}).get("class_labels", class_labels)
        except Exception:
            pass
    
    # Dataset statistics
    dataset_npz = os.path.join(root_dir, "data", f"{dataset_name}.npz")
    dataset_available = os.path.exists(dataset_npz)
    dataset_size_mb = round(os.path.getsize(dataset_npz) / (1024 * 1024), 2) if dataset_available else 0

    return {
        **training_state,
        "dataset_name": dataset_name,
        "dataset_available": dataset_available,
        "dataset_size_mb": dataset_size_mb,
        "dataset_classes": class_labels,
        "checkpoint": ckpt_info
    }

def run_background_train_task(epochs: int = 5):
    global training_state
    root_dir = get_root_dir()
    try:
        training_state["is_training"] = True
        training_state["status"] = "TRAINING"
        training_state["total_epochs"] = epochs
        training_state["message"] = f"Training 3D CNN on OrganMNIST3D for {epochs} epochs..."
        
        # Execute training pipeline
        import subprocess
        python_exe = os.path.join(root_dir, "venv", "Scripts", "python.exe")
        if not os.path.exists(python_exe):
            python_exe = "python"
            
        cmd = [python_exe, "main.py", "train"]
        result = subprocess.run(cmd, cwd=root_dir, capture_output=True, text=True)
        
        if result.returncode == 0:
            training_state["status"] = "COMPLETED"
            training_state["message"] = "Model trained successfully. Checkpoint saved."
        else:
            training_state["status"] = "ERROR"
            training_state["message"] = f"Training exited with code {result.returncode}: {result.stderr[:200]}"
    except Exception as e:
        training_state["status"] = "ERROR"
        training_state["message"] = str(e)
    finally:
        training_state["is_training"] = False

@router.post("/start", summary="Trigger Background 3D CNN Model Training")
def start_model_training(background_tasks: BackgroundTasks, epochs: int = 5):
    global training_state
    if training_state["is_training"]:
        return {"status": "ALREADY_RUNNING", "message": "Model training is already in progress."}

    background_tasks.add_task(run_background_train_task, epochs)
    training_state["is_training"] = True
    training_state["status"] = "STARTING"
    training_state["message"] = "Starting model training in background..."
    return {"status": "STARTED", "epochs": epochs, "message": "Model training initiated."}
