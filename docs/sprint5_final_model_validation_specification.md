# Sprint 5 — Final Model Validation Specification

**Project Title:** AI-Powered 3D Medical Image Reconstruction and Disease Analysis
**Project ID:** SKIT/DS/2023-2027/CSE-F-07
**Team Lead:** Mohammad Rafey
**Sprint Window:** 01-01-2027 to 31-01-2027

---

## 1. Overview & Objectives

Sprint 5 establishes a formal model validation suite to evaluate the final packaged 3D CNN model against known, predefined test cases. It verifies that:
- Disease predictions and class labels match ground truth expectations.
- Softmax confidence scores are produced correctly.
- Model predictions meet clinical confidence thresholds.
- Detailed validation metrics and case-by-case audit reports are saved to disk.

---

## 2. Validation Suite Architecture (`ModelValidator`)

Implemented in `src/validation/validator.py` (`ModelValidator`):
- Accepts input test cases (raw 3D volumes) and expected target labels.
- Preprocesses each test volume through the standard preprocessing pipeline.
- Performs inference via `DiseaseAnalyzer`.
- Assesses prediction accuracy, confidence score distributions, and threshold compliance.
- Outputs structured JSON validation report (`results/final_model_validation_report.json`).

---

## 3. Key Metrics Tracked

- **Overall Accuracy**: Fraction of test cases predicted correctly.
- **Average Confidence**: Mean softmax score across all test predictions.
- **Threshold Passing Rate**: Proportion of predictions meeting $\tau_{\text{threshold}} \ge 0.50$.
- **Per-case Detailed Audit**: Log of expected label vs. predicted label, raw probability distribution, and threshold flag.

---

## 4. CLI Subcommand Usage

Execute final validation:
```bash
python main.py validate --num-test-cases 10 --threshold 0.50
```
