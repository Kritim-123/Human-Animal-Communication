# Breed Classifier Worklog

This document records the breed-classifier work completed so far, the results we got, and why each process was implemented. It is intended to make the experiment history traceable from the repository.

## Starting Point

The project already had:

- `experiments/breed_classifier.py`
- `backend/data/processed/dog_audio_manifest.csv`
- audio files under `backend/data/raw/dogspeak_released`

The manifest contains five breed labels:

- `chihuahua`
- `gsd`
- `husky`
- `pitbull`
- `shibainu`

The full manifest has 55,669 rows. The class distribution is imbalanced:

| Breed | Clips |
| --- | ---: |
| husky | 19,379 |
| shibainu | 16,821 |
| chihuahua | 7,209 |
| gsd | 6,150 |
| pitbull | 6,110 |

The data also contains many clips per dog. Because of that, a random clip split can overstate model performance: clips from the same individual dog can appear in both train and test.

## Why We Changed the Workflow

The main goal was to make the breed classifier more honest and easier to improve.

The earlier random split answered:

> Can the model classify held-out clips?

The improved dog-grouped split answers:

> Can the model classify clips from dogs it has never seen?

That second question is much closer to what a real breed classifier needs.

## Implemented Workflow Improvements

## 1. Resumable Feature Extraction

`experiments/breed_classifier.py` now supports:

- cached feature CSV loading
- checkpointing feature extraction
- `--rebuild-features`
- `--checkpoint-every`
- `--max-examples-per-breed`

Why:

- The dataset is large.
- Feature extraction can take long enough that checkpointing matters.
- Balanced caps let us run comparable experiments quickly.

Main cached acoustic feature file:

```text
backend/data/processed/breed_features.csv
```

## 2. Dog-Grouped Evaluation

`experiments/breed_classifier.py` now supports:

- `--split-strategy dog-grouped`
- `--split-strategy random`

The default is dog-grouped.

Why:

- Prevents same-dog leakage between train and test.
- Gives a more realistic estimate of generalization.

## 3. Better Metrics and Reports

The classifier now saves:

- classification report
- confusion matrix
- JSON metrics
- model metadata
- train/test dog IDs
- top-2 accuracy
- top-3 accuracy
- balanced accuracy
- macro F1
- weighted F1

Main report directory:

```text
backend/data/models/reports/breed_classifier/
```

## 4. Multiple Classical Model Heads

`experiments/breed_classifier.py` now supports:

- `random_forest`
- `extra_trees`
- `logistic_regression`
- `hist_gradient_boosting`

Why:

- Random forest was a useful baseline, but not necessarily the best classifier for the feature table.
- Standardized logistic regression provides a strong linear baseline.
- Comparing model heads helps separate feature quality from classifier choice.

Current default:

```text
logistic_regression
```

## 5. YAMNet Embedding Experiment

Added:

```text
experiments/yamnet_breed_classifier.py
```

This script:

- loads YAMNet from TensorFlow Hub
- extracts YAMNet embeddings at 16 kHz
- pools embedding mean and standard deviation
- caches embeddings
- trains a classifier with the same dog-grouped evaluation workflow

Why:

- The improvement plan identified pretrained audio embeddings as the next major approach to test.
- YAMNet is a practical AudioSet-pretrained model that can be used as a feature extractor.

YAMNet needed TensorFlow, but the system Python was Python 3.14 and TensorFlow had no compatible wheel. To solve that, a local Python 3.11.15 runtime was installed under:

```text
.python311/
.venv311-yamnet/
```

Those directories are ignored by `.gitignore`.

YAMNet embedding cache:

```text
backend/data/processed/breed_yamnet_embeddings.csv
```

## 6. Fused Feature Experiment

Created a combined feature table from:

- MFCC/acoustic summary features
- YAMNet embeddings

Fused feature file:

```text
backend/data/processed/breed_fused_features.csv
```

Why:

- YAMNet might capture generic sound-event structure.
- MFCC/acoustic features might capture dog-specific acoustic cues.
- Fusion tests whether the feature sets are complementary.

## Commands Used for the Main Runs

Acoustic feature logistic baseline:

```bash
.venv/bin/python experiments/breed_classifier.py \
  --max-examples-per-breed 1000 \
  --split-strategy dog-grouped \
  --model-type logistic_regression
```

YAMNet logistic run:

```bash
.venv311-yamnet/bin/python experiments/yamnet_breed_classifier.py \
  --max-examples-per-breed 1000 \
  --checkpoint-every 100 \
  --model-type logistic_regression \
  --split-strategy dog-grouped
```

YAMNet random forest run:

```bash
.venv311-yamnet/bin/python experiments/yamnet_breed_classifier.py \
  --max-examples-per-breed 1000 \
  --checkpoint-every 100 \
  --model-type random_forest \
  --split-strategy dog-grouped \
  --model-path backend/data/models/breed_yamnet_random_forest.joblib \
  --report-dir backend/data/models/reports/breed_yamnet_random_forest
```

Fused feature run:

```bash
.venv/bin/python experiments/breed_classifier.py \
  --features-csv backend/data/processed/breed_fused_features.csv \
  --max-examples-per-breed 1000 \
  --split-strategy dog-grouped \
  --model-type logistic_regression \
  --model-path backend/data/models/breed_fused_logistic_regression.joblib \
  --report-dir backend/data/models/reports/breed_fused_logistic_regression
```

## Evaluation Setup

All main comparison runs used:

- 1,000 clips per breed
- 5,000 total clips
- dog-grouped holdout split
- 3,616 train rows
- 1,384 test rows
- 105 train dogs
- 35 test dogs

## Results

| Feature Set | Model | Accuracy | Balanced Accuracy | Macro F1 | Top-2 | Top-3 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MFCC/acoustic stats | Logistic regression | 0.419 | 0.434 | 0.411 | 0.645 | 0.793 |
| MFCC/acoustic stats | Hist gradient boosting | 0.413 | 0.435 | 0.403 | 0.626 | 0.743 |
| MFCC/acoustic stats | Random forest | 0.400 | 0.444 | 0.395 | 0.583 | 0.721 |
| MFCC/acoustic stats | Extra trees | 0.368 | 0.401 | 0.360 | 0.583 | 0.707 |
| MFCC + YAMNet fusion | Logistic regression | 0.348 | 0.372 | 0.340 | 0.589 | 0.752 |
| YAMNet embeddings | Random forest | 0.325 | 0.354 | 0.320 | 0.548 | 0.710 |
| YAMNet embeddings | Logistic regression | 0.299 | 0.304 | 0.288 | 0.511 | 0.702 |

Full comparison CSV:

```text
backend/data/models/reports/breed_all_model_comparison.csv
```

## Interpretation

The best current model is:

```text
MFCC/acoustic stats + logistic regression
```

Main model artifact:

```text
backend/data/models/breed_classifier.joblib
```

YAMNet was implemented and tested, but it did not beat the handcrafted acoustic-statistics baseline under dog-grouped evaluation.

This does not mean YAMNet is useless. It means generic AudioSet embeddings, used frozen and pooled simply, were not better for this specific breed-classification setup. Breed cues may be more tied to fine-grained acoustic structure, dog identity variation, clip quality, or dataset-specific recording conditions than to generic audio event semantics.

## Why We Kept the Acoustic Logistic Model as Main

The acoustic logistic model has:

- best top-1 accuracy
- best macro F1
- best top-2 accuracy
- best top-3 accuracy
- much smaller feature set than YAMNet
- simpler runtime requirements

Current main model metadata:

```text
model_version: dogbridge_breed_logistic_regression_v2
accuracy: 0.4190751445086705
top_3_accuracy: 0.7926300578034682
```

## Important Generated Artifacts

These are ignored by git because `backend/data/processed/*` and `backend/data/models/*` are ignored:

```text
backend/data/processed/breed_features.csv
backend/data/processed/breed_yamnet_embeddings.csv
backend/data/processed/breed_fused_features.csv
backend/data/models/breed_classifier.joblib
backend/data/models/breed_yamnet_classifier.joblib
backend/data/models/breed_yamnet_random_forest.joblib
backend/data/models/breed_fused_logistic_regression.joblib
backend/data/models/reports/breed_all_model_comparison.csv
```

## Code and Documentation Added

Tracked files from this work:

```text
.gitignore
experiments/breed_classifier.py
experiments/yamnet_breed_classifier.py
docs/BREED_CLASSIFIER_IMPROVEMENT_PLAN.md
docs/BREED_CLASSIFIER_EXPERIMENT_RESULTS.md
docs/BREED_CLASSIFIER_WORKLOG.md
```

## Remaining Recommended Next Steps

1. Add clip-quality filtering.
2. Add dog-grouped cross-validation instead of a single holdout split.
3. Try larger per-breed caps.
4. Add error analysis by dog and breed pair.
5. Try a supervised log-mel spectrogram CNN.
6. If compute allows, fine-tune an audio model rather than using frozen YAMNet embeddings.
