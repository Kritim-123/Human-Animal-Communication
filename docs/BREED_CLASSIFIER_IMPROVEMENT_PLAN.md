# Breed Classifier Improvement Plan

This document lists approaches that can improve the current breed-classification workflow.

Current workflow:

- Manifest: `backend/data/processed/dog_audio_manifest.csv`
- Audio: `backend/data/raw/dogspeak_released`
- Experiment script: `experiments/breed_classifier.py`
- Current model: handcrafted acoustic features plus `RandomForestClassifier`
- Current trained subset: balanced sample of 1,000 clips per breed
- Current labels: `chihuahua`, `gsd`, `husky`, `pitbull`, `shibainu`

## Highest Priority Improvements

## 1. Split by Dog ID Instead of Random Clip Split

The current workflow uses a random clip-level train/test split. This can overestimate performance because clips from the same dog may appear in both training and test sets.

Better approach:

- Use `dog_id` as a group.
- Keep every clip from a dog entirely in train or test.
- Prefer `StratifiedGroupKFold` for cross-validation.
- Use `GroupShuffleSplit` for a single train/test split.

Why this helps:

- Measures whether the model generalizes to new dogs.
- Reduces leakage from individual dog identity, microphone, room acoustics, or recording style.
- Gives a more honest estimate of breed recognition performance.

Implementation idea:

- Add `--split-strategy random|dog-grouped`.
- Add `--cv-folds 5`.
- Report both random-split and dog-grouped results during experimentation.

Priority: very high.

## 2. Add Better Evaluation Metrics

Accuracy alone hides where the model fails.

Add:

- Macro F1.
- Per-breed precision, recall, and F1.
- Confusion matrix.
- Top-2 and top-3 accuracy.
- Balanced accuracy.
- Cross-validation mean and standard deviation.

Why this helps:

- Shows whether one breed is being favored.
- Shows which breeds are confused with each other.
- Makes it easier to compare MFCC, YAMNet, PANNs, and AST experiments fairly.

Implementation idea:

- Save reports to `backend/data/models/reports/`.
- Write:
  - `breed_classification_report.txt`
  - `breed_confusion_matrix.csv`
  - `breed_metrics.json`

Priority: very high.

## 3. Use Pretrained Audio Embeddings

The current model uses handcrafted features: MFCC, pitch, RMS, zero-crossing rate, and spectral features. These are useful, but modern audio classifiers often perform better with pretrained embeddings.

Candidate embedding models:

- YAMNet
- PANNs
- Audio Spectrogram Transformer
- HuBERT or wav2vec-style self-supervised audio models

Recommended first upgrade:

- Extract YAMNet embeddings for every clip.
- Average or max-pool frame embeddings per clip.
- Train a lightweight classifier on top:
  - logistic regression
  - linear SVM
  - random forest
  - gradient boosting

Why this helps:

- Pretrained audio models learn richer sound patterns than simple summary features.
- YAMNet is practical and relatively easy to add.
- Embeddings can be cached just like the current feature CSV.

Implementation idea:

- Add `experiments/yamnet_breed_classifier.py`.
- Cache embeddings to `backend/data/processed/breed_yamnet_embeddings.parquet`.
- Train the same classifier/evaluation interface as `breed_classifier.py`.

Priority: very high.

## 4. Use More of the Dataset Without Letting Large Classes Dominate

The full manifest is imbalanced:

- `husky`: 19,379 clips
- `shibainu`: 16,821 clips
- `chihuahua`: 7,209 clips
- `gsd`: 6,150 clips
- `pitbull`: 6,110 clips

The first trained baseline used 1,000 clips per breed for speed.

Better approach:

- Keep balanced batches during training.
- Increase the per-breed cap gradually:
  - 1,000
  - 2,500
  - 5,000
  - all available clips with class weighting
- Compare results under dog-grouped validation.

Why this helps:

- More data may improve generalization.
- Balanced sampling prevents the model from over-learning the largest breeds.
- Gradual caps make runtime manageable.

Priority: high.

## 5. Improve Audio Cleaning and Segmentation

The current feature extraction runs directly on each clip.

Add preprocessing:

- Convert to mono.
- Resample consistently.
- Trim leading and trailing silence.
- Normalize loudness.
- Remove or flag extremely short clips.
- Remove or flag clips with low signal-to-noise ratio.
- Optionally detect and exclude human speech-heavy clips.

Why this helps:

- Reduces noise in feature extraction.
- Avoids training on silence or background noise.
- Helps the model focus on dog vocalizations.

Implementation idea:

- Add `experiments/audio_quality.py`.
- Write quality fields back to a processed metadata file:
  - `duration_seconds`
  - `rms_mean`
  - `silence_fraction`
  - `clipped_audio`
  - `quality_pass`

Priority: high.

## Modeling Improvements

## 6. Compare Multiple Classical Models

The random forest is a good baseline, but it should not be the only classical model.

Try:

- Logistic regression with standardized features.
- Linear SVM.
- RBF SVM on smaller subsets.
- ExtraTreesClassifier.
- HistGradientBoostingClassifier.
- CalibratedClassifierCV for better probabilities.

Why this helps:

- Some models may generalize better on grouped splits.
- Linear models are useful sanity checks for embedding quality.
- Calibrated probabilities make confidence thresholds more reliable.

Priority: medium.

## 7. Add Hyperparameter Search

Current random forest settings are fixed.

Tune:

- `n_estimators`
- `max_depth`
- `min_samples_leaf`
- `max_features`
- `class_weight`

Use:

- `RandomizedSearchCV`
- dog-grouped cross-validation
- macro F1 as the main score

Why this helps:

- May improve the baseline without changing features.
- Makes model comparisons less dependent on arbitrary defaults.

Priority: medium.

## 8. Add Spectrogram CNN Experiments

Instead of extracting summary features, train on log-mel spectrogram images.

Approach:

- Convert clips to fixed-length log-mel spectrograms.
- Pad or crop clips to a fixed duration.
- Train a small CNN.
- Use data augmentation:
  - time masking
  - frequency masking
  - random gain
  - small time shifts
  - background noise mixing

Why this helps:

- Preserves time-frequency structure lost by summary statistics.
- Can learn bark shape and temporal patterns.

Tradeoff:

- More code and training complexity.
- Needs careful validation to avoid overfitting.

Priority: medium.

## 9. Fine-Tune a Pretrained Audio Model

After embedding experiments, fine-tuning may improve performance.

Candidate models:

- PANNs
- AST
- wav2vec2-style audio encoders
- HuBERT-style self-supervised models

Why this helps:

- Lets the model adapt to dog vocalization audio rather than generic AudioSet classes.
- May outperform frozen embeddings.

Tradeoff:

- Requires more compute.
- Requires a stronger training pipeline.
- Easier to overfit if dog-level validation is not used.

Priority: medium to low until grouped evaluation is in place.

## Data Improvements

## 10. Analyze Per-Dog Distribution

The dataset contains many clips per dog. Some dogs may dominate their breed.

Add analysis:

- clips per dog
- dogs per breed
- duration per dog
- recording quality per dog
- train/test dog distribution

Why this helps:

- Prevents one dog from dominating model behavior.
- Reveals whether low performance is due to too few individual dogs.

Priority: high.

## 11. Add Metadata Features Carefully

The manifest currently includes breed, sex, dog ID, folder, and file path.

Possible metadata:

- sex
- duration
- clip source
- recording quality features

Use caution:

- Do not include `dog_id`, `folder_name`, or filename-derived breed tokens as model features.
- These leak labels or identity.

Why this helps:

- Some metadata may improve robustness.
- Avoiding leakage keeps evaluation honest.

Priority: medium.

## 12. Add Error Analysis Workflow

After each run, inspect the worst errors.

Track:

- confidently wrong predictions
- low-confidence correct predictions
- breed pairs with high confusion
- short clips with wrong predictions
- noisy clips with wrong predictions

Why this helps:

- Shows whether the problem is model architecture, data quality, or label ambiguity.
- Helps decide whether to clean data or change models.

Priority: high.

## 13. Add a Human-Review Sampling Set

Create a small reviewed validation subset.

Approach:

- Sample clips from every breed and dog.
- Listen to a few hundred clips.
- Mark:
  - clear dog vocalization
  - background noise
  - human speech
  - silence
  - uncertain label

Why this helps:

- Reveals label noise.
- Produces a trusted evaluation subset.

Priority: medium.

## Workflow Improvements

## 14. Make Experiments Reproducible

Every model run should save:

- command-line args
- git commit if available
- manifest path
- feature path
- model path
- random seed
- train/test split IDs
- metrics
- package versions

Why this helps:

- Makes it possible to compare experiments later.
- Prevents losing track of which model produced which score.

Priority: high.

## 15. Use a Consistent Experiment Output Directory

Suggested layout:

```text
backend/data/models/breed/
  random_forest_mfcc/
    model.joblib
    metrics.json
    classification_report.txt
    confusion_matrix.csv
    run_config.json
  yamnet_logreg/
    model.joblib
    metrics.json
    classification_report.txt
    confusion_matrix.csv
    run_config.json
```

Why this helps:

- Keeps experiments organized.
- Makes it easier to promote one model into the backend.

Priority: medium.

## 16. Add Model Registry Metadata

Each saved model bundle should include:

- model name
- model version
- labels
- feature names
- feature extractor version
- sample rate
- validation strategy
- metrics
- training date

Why this helps:

- Backend inference can verify it is loading the right model.
- Prevents mismatches between feature extraction and model input.

Priority: medium.

## 17. Add Prediction Confidence Handling

Breed classification should expose uncertainty.

Add:

- top-k predictions
- confidence threshold
- `unknown` or `uncertain` when confidence is low
- probability calibration

Why this helps:

- A wrong high-confidence breed prediction is more damaging than an honest uncertain result.
- Top-k predictions may be more useful than a single label.

Priority: medium.

## 18. Integrate the Breed Model Into Backend Inference

The current trained breed model lives as an experiment artifact.

Add:

- backend service method for breed prediction
- endpoint or internal method for breed classifier
- model-loading config
- response schema with top-k probabilities

Why this helps:

- Makes the model usable by the app.
- Keeps experiment code separate from production inference.

Priority: medium.

## Suggested Roadmap

## Phase 1: Honest Baseline

- Add dog-grouped split.
- Add confusion matrix and top-k metrics.
- Save metrics and run config.
- Re-run current MFCC random forest.

Goal:

- Know the true baseline under new-dog evaluation.

## Phase 2: Strong Embedding Baseline

- Add YAMNet embedding extraction.
- Cache embeddings.
- Train logistic regression and random forest on embeddings.
- Compare against MFCC baseline using the same dog-grouped split.

Goal:

- Establish a stronger non-deep-training baseline.

## Phase 3: Data Quality

- Add audio quality analysis.
- Filter silence and very short clips.
- Compare filtered vs unfiltered runs.
- Run error analysis.

Goal:

- Improve training data quality and understand failure modes.

## Phase 4: Neural Models

- Train a log-mel CNN.
- Try PANNs or AST embeddings.
- Consider fine-tuning only after grouped validation is stable.

Goal:

- Move beyond lightweight classifiers if embeddings are not enough.

## Phase 5: Product Integration

- Add backend inference support.
- Add model metadata checks.
- Return top-k predictions and uncertainty.

Goal:

- Make the breed classifier available to the app in a responsible way.

## Recommended Next Task

Implement dog-grouped evaluation first.

This is the best next step because it changes the question from:

> Can the model classify held-out clips?

to:

> Can the model classify clips from dogs it has never seen?

That is the metric that matters for a real breed classifier.

## References

- TensorFlow YAMNet tutorial: https://www.tensorflow.org/hub/tutorials/yamnet
- PANNs paper: https://arxiv.org/abs/1912.10211
- Audio Spectrogram Transformer paper: https://arxiv.org/abs/2104.01778
- scikit-learn GroupShuffleSplit: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html
- scikit-learn StratifiedGroupKFold: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
- Canine HuBERT paper: https://arxiv.org/abs/2402.15985
- Dog vocalization acoustic correlation paper: https://arxiv.org/abs/2309.13085
