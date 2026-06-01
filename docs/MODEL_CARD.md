# DogBridge Baseline Model Card

## Purpose

The DogBridge baseline model estimates a dog's likely intent from a short audio clip. It is designed for personalized feedback loops where owners confirm or correct predictions.

## Input

- Dog audio clip.
- Optional contextual metadata stored with the clip.
- MVP feature extraction uses audio only.

## Output

- Predicted likely intent label.
- Confidence score.
- Top likely labels.
- Safety message for `stress_discomfort`.

## Model

The MVP baseline extracts MFCC, chroma, spectral centroid, spectral bandwidth, zero crossing rate, and RMS energy statistics. It trains a `RandomForestClassifier`.

## Limitations

- It requires real labeled dog audio.
- It may perform poorly with small or imbalanced datasets.
- It does not understand language.
- It does not account for all environmental factors.
- It returns `unknown` when confidence is below `0.55`.
- Reported confidence is a model probability estimate, not a guarantee.

## Ethical Warning

DogBridge estimates possible intent. It does not literally translate dog language and should never be presented as absolute truth.

## Safety Warning

`stress_discomfort` predictions are not veterinary diagnosis. Monitor your dog. If this is unusual or persistent, contact a veterinarian.

