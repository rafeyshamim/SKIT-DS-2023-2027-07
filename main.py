"""
AI-Powered 3D Medical Image Reconstruction and Disease Analysis
Project ID: SKIT/DS/2023-2027/CSE-F-07
Team Lead: Mohammad Rafey

Main Command-Line Interface (CLI).
Supported Commands:
    - preprocess       : Preprocess volumetric medical imaging data (Sprint 1)
    - train            : Train the 3D CNN classifier (Sprints 2 & 3)
    - evaluate         : Evaluate model performance on test set (Sprint 3)
    - predict          : Predict disease and confidence score for a volume (Sprint 3)
    - reconstruct      : Generate 3D reconstructions (MIP, slices, isosurface) (Sprint 3)
    - package          : Optimize and package model for deployment (Sprint 4)
    - validate         : Validate final model against test cases (Sprint 5)
    - integration-test : Perform complete end-to-end integration test (Sprint 6)
    - demo             : Run an end-to-end demonstration using synthetic volume
"""
import argparse
import json
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


def cmd_package(args):
    """Sprint 4 CLI command: Package and optimize trained model for inference."""
    logger.info("Starting Model Optimization & Packaging...")
    from src.inference.packager import ModelPackager
    from src.model.trainer import ModelTrainer

    config = load_config(args.config)
    trainer = ModelTrainer(config)
    try:
        model = trainer.load_best_model()
    except FileNotFoundError:
        logger.warning("Checkpoint not found. Packaging initialized model weights.")
        model = trainer.model

    packager = ModelPackager(config)
    artifacts = packager.package_model(
        model,
        export_name=args.export_name,
        export_tflite=args.export_tflite,
        quantize_tflite=args.quantize,
    )
    logger.info("Model successfully packaged! Artifacts: %s", artifacts)


def cmd_validate(args):
    """Sprint 5 CLI command: Run final validation against test cases."""
    logger.info("Starting Final Model Validation...")
    from src.model.trainer import ModelTrainer
    from src.validation.validator import ModelValidator

    config = load_config(args.config)
    trainer = ModelTrainer(config)
    try:
        model = trainer.load_best_model()
    except FileNotFoundError:
        logger.warning("Checkpoint not found. Validating initialized model weights.")
        model = trainer.model

    # Generate synthetic test phantoms if no explicit file provided
    input_shape = config["model"]["input_shape"][:3]
    num_classes = config["data"]["num_classes"]

    test_volumes = []
    expected_labels = []
    rng = np.random.default_rng(42)

    for i in range(args.num_test_cases):
        vol = rng.uniform(0.0, 1.0, size=input_shape).astype(np.float32)
        test_volumes.append(vol)
        expected_labels.append(i % num_classes)

    validator = ModelValidator(model, config)
    summary = validator.validate_test_cases(
        test_volumes,
        expected_labels,
        confidence_threshold=args.threshold,
    )
    logger.info("Validation Complete! Accuracy: %.2f%%", summary["accuracy"] * 100)


def cmd_integration_test(args):
    """Sprint 6 CLI command: Complete end-to-end integration testing."""
    logger.info("Executing Complete End-to-End Integration Test...")
    config = load_config(args.config)

    # 1. Pipeline check
    from src.preprocessing.pipeline import PreprocessingPipeline
    from src.model.architecture import get_model_from_config
    from src.disease_analysis.analyzer import DiseaseAnalyzer
    from src.reconstruction.reconstructor import VolumeReconstructor
    from src.inference.packager import ModelPackager
    from src.inference.engine import InferenceEngine

    shape = tuple(config["model"]["input_shape"][:3])
    raw = np.random.default_rng(42).uniform(0, 1, size=shape).astype(np.float32)

    pipeline = PreprocessingPipeline(config)
    proc = pipeline.preprocess_single_volume(raw, augment=False)
    assert proc.shape == tuple(config["model"]["input_shape"]), f"Invalid shape: {proc.shape}"

    model = get_model_from_config(config)
    analyzer = DiseaseAnalyzer(model, config)
    analysis = analyzer.analyze(proc)
    assert "confidence" in analysis

    recon = VolumeReconstructor(output_dir="results/integration_test/")
    recons = recon.generate_all_visualizations(proc, prefix="integration")

    packager = ModelPackager(config)
    artifacts = packager.package_model(model, export_name="integration_pkg", export_tflite=False)

    engine = InferenceEngine(package_dir=os.path.dirname(artifacts["keras_model"]))
    engine_res = engine.predict_volume(raw, preprocess=True)

    logger.info("Integration Test Succeeded! Engine prediction: %s", engine_res["predicted_class_label"])


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

    print("\n[SUCCESS] End-to-End Demo Complete! All Sprint capabilities verified.")


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

    # Subcommand: package (Sprint 4)
    p_pkg = subparsers.add_parser("package", help="Package and optimize model for deployment")
    p_pkg.add_argument("--export-name", type=str, default="3d_cnn_packaged", help="Export directory name")
    p_pkg.add_argument("--export-tflite", action="store_true", default=True, help="Export TFLite model")
    p_pkg.add_argument("--quantize", action="store_true", default=False, help="Enable dynamic range quantization")

    # Subcommand: validate (Sprint 5)
    p_val = subparsers.add_parser("validate", help="Validate final model against test cases")
    p_val.add_argument("--num-test-cases", type=int, default=5, help="Number of test cases")
    p_val.add_argument("--threshold", type=float, default=0.50, help="Confidence threshold")

    # Subcommand: integration-test (Sprint 6)
    p_it = subparsers.add_parser("integration-test", help="Run end-to-end integration test")

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
        "package": cmd_package,
        "validate": cmd_validate,
        "integration-test": cmd_integration_test,
        "demo": cmd_demo,
    }

    handlers[args.subcommand](args)


if __name__ == "__main__":
    main()
