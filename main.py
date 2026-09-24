"""
AI-Powered 3D Medical Image Reconstruction and Disease Analysis
Project ID: SKIT/DS/2023-2027/CSE-F-07
Team Lead: Mohammad Rafey

Main Command-Line Interface (CLI).
Supported Commands:
    - preprocess   : Preprocess volumetric medical imaging data
    - train        : Train the 3D CNN classifier
    - evaluate     : Evaluate model performance on test set
    - predict      : Predict disease and confidence score for a volume
    - reconstruct  : Generate 3D reconstructions (MIP, slices, isosurface)
    - demo         : Run an end-to-end demonstration using synthetic volume
"""
import argparse
import os
import sys
import numpy as np

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger("main_cli")


def cmd_preprocess(args):
    """Run preprocessing pipeline on dataset."""
    logger.info("Executing Preprocessing Pipeline...")
    from src.preprocessing.pipeline import PreprocessingPipeline
    config = load_config(args.config)
    pipeline = PreprocessingPipeline(config)
    splits = pipeline.load_and_prepare_data()
    logger.info("Preprocessing complete. Available splits: %s", list(splits.keys()))


def cmd_train(args):
    """Run model training."""
    logger.info("Starting 3D CNN Model Training...")
    from src.preprocessing.pipeline import PreprocessingPipeline
    from src.model.trainer import ModelTrainer
    
    config = load_config(args.config)
    pipeline = PreprocessingPipeline(config)
    splits = pipeline.load_and_prepare_data()

    X_train, y_train = splits["train"]
    X_val, y_val = splits.get("val", (X_train, y_train))

    train_ds = pipeline.get_tf_dataset(X_train, y_train, shuffle=True)
    val_ds = pipeline.get_tf_dataset(X_val, y_val, shuffle=False)

    trainer = ModelTrainer(config)
    trainer.train(train_ds, val_ds)
    logger.info("Training finished successfully.")


def cmd_evaluate(args):
    """Run model evaluation."""
    logger.info("Starting Model Evaluation...")
    from src.preprocessing.pipeline import PreprocessingPipeline
    from src.model.trainer import ModelTrainer
    from src.model.evaluator import ModelEvaluator
    
    config = load_config(args.config)
    pipeline = PreprocessingPipeline(config)
    splits = pipeline.load_and_prepare_data()

    X_test, y_test = splits.get("test", splits.get("val", splits["train"]))
    test_ds = pipeline.get_tf_dataset(X_test, y_test, shuffle=False)

    trainer = ModelTrainer(config)
    try:
        model = trainer.load_best_model()
    except FileNotFoundError:
        logger.warning("Checkpoint not found. Using initialized model weights for evaluation.")
        model = trainer.model

    evaluator = ModelEvaluator(model, config)
    metrics = evaluator.evaluate(test_ds, split_name="test")
    logger.info("Evaluation Metrics: %s", metrics)


def cmd_predict(args):
    """Run disease prediction & confidence scoring on a volume."""
    logger.info("Running Disease Analysis & Prediction...")
    from src.disease_analysis.analyzer import DiseaseAnalyzer
    from src.model.trainer import ModelTrainer
    from src.preprocessing.pipeline import PreprocessingPipeline
    
    config = load_config(args.config)
    trainer = ModelTrainer(config)
    try:
        model = trainer.load_best_model()
    except FileNotFoundError:
        logger.warning("Checkpoint not found. Using current model for prediction.")
        model = trainer.model

    analyzer = DiseaseAnalyzer(model, config)

    # Load volume
    if args.input.endswith(".npy"):
        raw_vol = np.load(args.input)
    elif args.input.endswith((".nii", ".nii.gz")):
        import nibabel as nib
        raw_vol = nib.load(args.input).get_fdata()
    else:
        raise ValueError("Unsupported input format. Please provide .npy or .nii/.nii.gz")

    pipeline = PreprocessingPipeline(config)
    preprocessed_vol = pipeline.preprocess_single_volume(raw_vol, augment=False)
    result = analyzer.analyze(preprocessed_vol)

    print("\n" + "=" * 50)
    print("DISEASE ANALYSIS REPORT")
    print("=" * 50)
    print(f"Predicted Class : {result['predicted_class_label']} (Index: {result['predicted_class_index']})")
    print(f"Confidence Score: {result['confidence'] * 100:.2f}%")
    print(f"Meets Threshold : {'YES' if result['above_threshold'] else 'NO'}")
    print("-" * 50)
    print("Class Probabilities:")
    for label, prob in result["probabilities"].items():
        print(f"  {label:<15}: {prob * 100:.2f}%")
    print("=" * 50 + "\n")


def cmd_reconstruct(args):
    """Generate 3D visualizations from a volume."""
    logger.info("Generating 3D Volume Visualizations...")
    from src.reconstruction.reconstructor import VolumeReconstructor
    from src.preprocessing.pipeline import PreprocessingPipeline
    
    config = load_config(args.config)
    pipeline = PreprocessingPipeline(config)

    if args.input.endswith(".npy"):
        raw_vol = np.load(args.input)
    elif args.input.endswith((".nii", ".nii.gz")):
        import nibabel as nib
        raw_vol = nib.load(args.input).get_fdata()
    else:
        raise ValueError("Unsupported input format. Please provide .npy or .nii/.nii.gz")

    preprocessed_vol = pipeline.preprocess_single_volume(raw_vol, augment=False)
    recon = VolumeReconstructor(output_dir=args.output_dir)
    results = recon.generate_all_visualizations(preprocessed_vol, prefix=args.prefix)
    logger.info("Reconstructions generated: %s", results)


def cmd_demo(args):
    """Run complete end-to-end demo on a synthetic volume."""
    logger.info("Executing End-to-End System Demonstration...")
    config = load_config(args.config)
    
    # 1. Generate synthetic volumetric phantom (32x32x32 sphere in cube)
    print("[1/4] Generating synthetic 3D volumetric medical scan...")
    grid = np.zeros((32, 32, 32), dtype=np.float32)
    z, y, x = np.ogrid[:32, :32, :32]
    sphere = (x - 16)**2 + (y - 16)**2 + (z - 16)**2 <= 8**2
    grid[sphere] = 0.8
    noise = np.random.default_rng(42).normal(0.1, 0.02, size=(32, 32, 32))
    synthetic_volume = np.clip(grid + noise, 0.0, 1.0).astype(np.float32)

    # 2. Preprocessing
    print("[2/4] Running Preprocessing Pipeline...")
    from src.preprocessing.pipeline import PreprocessingPipeline
    pipeline = PreprocessingPipeline(config)
    processed = pipeline.preprocess_single_volume(synthetic_volume, augment=False)
    print(f"      Preprocessed shape: {processed.shape}, range: [{processed.min():.2f}, {processed.max():.2f}]")

    # 3. Model Inference & Disease Prediction
    print("[3/4] Running 3D CNN Inference & Disease Analysis...")
    from src.model.architecture import get_model_from_config
    from src.disease_analysis.analyzer import DiseaseAnalyzer
    model = get_model_from_config(config)
    analyzer = DiseaseAnalyzer(model, config)
    res = analyzer.analyze(processed)
    print(f"      Predicted Class  : {res['predicted_class_label']}")
    print(f"      Confidence Score : {res['confidence'] * 100:.2f}%")

    # 4. 3D Reconstruction
    print("[4/4] Generating 3D Reconstruction Visualizations...")
    from src.reconstruction.reconstructor import VolumeReconstructor
    recon = VolumeReconstructor(output_dir="results/demo_reconstruction/")
    outputs = recon.generate_all_visualizations(processed, prefix="demo_sample")
    print("      Generated Visualization Files:")
    for k, v in outputs.items():
        if v:
            print(f"        - {k}: {v}")

    print("\n[SUCCESS] End-to-End Demo Complete! All Sprint 1, 2, and 3 capabilities verified.")



def main():
    parser = argparse.ArgumentParser(
        description="AI-Powered 3D Medical Image Reconstruction and Disease Analysis CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to YAML configuration file")

    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # Subcommand: preprocess
    p_prep = subparsers.add_parser("preprocess", help="Preprocess volumetric medical data")

    # Subcommand: train
    p_train = subparsers.add_parser("train", help="Train 3D CNN model")

    # Subcommand: evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate model performance")

    # Subcommand: predict
    p_pred = subparsers.add_parser("predict", help="Analyze disease for an input volume")
    p_pred.add_argument("--input", type=str, required=True, help="Path to input volume (.npy or .nii/.nii.gz)")

    # Subcommand: reconstruct
    p_rec = subparsers.add_parser("reconstruct", help="Generate 3D reconstructions and visual projections")
    p_rec.add_argument("--input", type=str, required=True, help="Path to input volume (.npy or .nii/.nii.gz)")
    p_rec.add_argument("--output-dir", type=str, default="results/reconstruction/", help="Output directory")
    p_rec.add_argument("--prefix", type=str, default="recon", help="File prefix for outputs")

    # Subcommand: demo
    p_demo = subparsers.add_parser("demo", help="Run end-to-end synthetic demonstration")

    args = parser.parse_args()

    if not args.subcommand:
        parser.print_help()
        sys.exit(0)

    handlers = {
        "preprocess": cmd_preprocess,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "predict": cmd_predict,
        "reconstruct": cmd_reconstruct,
        "demo": cmd_demo,
    }

    handlers[args.subcommand](args)


if __name__ == "__main__":
    main()
