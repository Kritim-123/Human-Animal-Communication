# Breed Classifier Experiment Results

This document summarizes the dog-grouped breed-classifier experiments run on a balanced 5,000-clip subset.

## Evaluation Setup

- Dataset: `backend/data/processed/dog_audio_manifest.csv`
- Audio root: `backend/data/raw/dogspeak_released`
- Classes: `chihuahua`, `gsd`, `husky`, `pitbull`, `shibainu`
- Sampling: 1,000 clips per breed
- Split: dog-grouped holdout
- Train rows: 3,616
- Test rows: 1,384
- Train dogs: 105
- Test dogs: 35

Dog-grouped evaluation keeps all clips from a dog entirely in train or test. This is stricter than random clip splitting and better measures generalization to unseen dogs.

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

## Current Best Model

The best current model is still the MFCC/acoustic-statistics logistic regression model.

Main artifact:

```text
backend/data/models/breed_classifier.joblib
```

Main reports:

```text
backend/data/models/reports/breed_classifier/
```

## YAMNet Artifacts

YAMNet is now implemented and tested.

Script:

```text
experiments/yamnet_breed_classifier.py
```

Embedding cache:

```text
backend/data/processed/breed_yamnet_embeddings.csv
```

YAMNet logistic model:

```text
backend/data/models/breed_yamnet_classifier.joblib
```

YAMNet random forest model:

```text
backend/data/models/breed_yamnet_random_forest.joblib
```

## Takeaway

YAMNet embeddings did not improve breed classification on this dog-grouped split. This suggests that generic AudioSet embeddings may not capture the breed-specific acoustic cues needed here, or that the current labels/split emphasize dog-to-dog variation more than generic sound-event features.

The most useful next steps are:

- add clip quality filtering
- add dog-grouped cross-validation instead of one holdout split
- try larger per-breed caps
- try supervised spectrogram CNNs or fine-tuned audio models
- inspect errors by dog and breed pair
