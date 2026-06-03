# DogBridge

DogBridge is an MVP for personalized dog-human communication research. It records dog audio, stores owner-provided context and labels, trains a simple per-dataset baseline, predicts a dog's likely intent for new clips, and lets the owner confirm or correct the result.

DogBridge is not a dog-to-English translator. It estimates possible intent from patterns in audio and context, then improves through owner feedback.

## MVP Intent Labels

- `outside_bathroom`
- `food_water`
- `play`
- `attention`
- `stress_discomfort`
- `unknown`

Low-confidence predictions are returned as `unknown`. Predictions must be treated as suggestions, never absolute truth.

## Run The Backend

```bash
cd dogbridge/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the FastAPI docs.

You can also run with Docker Compose:

```bash
cd dogbridge
docker compose up --build
```

## Basic API Flow

Create a dog:

```bash
curl -X POST http://127.0.0.1:8000/dogs \
  -H "Content-Type: application/json" \
  -d '{"name":"Milo","breed":"Mixed","age":4}'
```

Upload a clip:

```bash
curl -X POST http://127.0.0.1:8000/clips \
  -F dog_id=1 \
  -F location_context=door \
  -F situation_context=before_walk \
  -F owner_label=outside_bathroom \
  -F outcome_label=outside_bathroom \
  -F notes="Whining near the door" \
  -F file=@/path/to/real-dog-audio.wav
```

Predict a likely intent:

```bash
curl -X POST http://127.0.0.1:8000/predict/1
```

Confirm or correct a prediction:

```bash
curl -X POST http://127.0.0.1:8000/confirm/1 \
  -H "Content-Type: application/json" \
  -d '{"confirmed_correct":false,"corrected_label":"attention","outcome_label":"attention"}'
```

## Feature Extraction

Use real audio clips. The dummy data script creates metadata only; it does not create fake dog audio.

```bash
cd dogbridge/backend
python -m scripts.extract_features --input-dir data/raw --output-csv data/processed/features.csv
```

If you have a metadata CSV with `file_path` and `owner_label`, pass it with:

```bash
python -m scripts.extract_features \
  --input-dir data/raw \
  --metadata-csv data/processed/metadata.csv \
  --output-csv data/processed/features.csv
```

## Train The Baseline

Train from uploaded clips in SQLite:

```bash
cd dogbridge/backend
python -m scripts.train_baseline
```

Or train from an extracted feature CSV:

```bash
python -m scripts.train_baseline --features-csv data/processed/features.csv
```

The model is saved to `backend/data/models/dogbridge_baseline.joblib`.

## Public Dog Audio Datasets

Public datasets can help DogBridge learn general dog audio structure, but they should not be treated as direct labels for owner intent. See [PUBLIC_DATASETS.md](docs/PUBLIC_DATASETS.md).

For a start-to-finish research workflow, use [DogSpeak_YAMNet_Pipeline.ipynb](notebooks/DogSpeak_YAMNet_Pipeline.ipynb). It downloads DogSpeak from Hugging Face, builds a public manifest, extracts YAMNet embeddings, and trains quick sanity-check classifiers.

After downloading DogSpeak locally, build a separate public-data manifest:

```bash
cd dogbridge/backend
python -m scripts.build_public_manifest \
  --dataset dogspeak \
  --input-metadata data/public/audio/DogSpeak/metadata.csv \
  --audio-root data/public/audio/DogSpeak \
  --output-csv data/public/manifests/dogspeak_manifest.csv
```

Downloaded public audio stays ignored by Git. Public manifests are tracked only when intentionally added.

## Run The Mobile Skeleton

```bash
cd dogbridge/mobile
npm install
npm start
```

The Expo app contains screens for dog profile setup, real audio recording, context labeling, clip upload, prediction review, confirmation, and basic dataset progress.

For simulator or local web development, the default API URL is `http://127.0.0.1:8000`.

For a physical phone on your Wi-Fi network, start the backend on all interfaces and point Expo at your computer's LAN address:

```bash
cd dogbridge/backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

cd ../mobile
EXPO_PUBLIC_DOGBRIDGE_API_URL=http://YOUR_LAN_IP:8000 npm run start:lan
```

## Current Limitations

- Requires real dog audio before training.
- Baseline model uses MFCC/chroma/spectral features, not deep learning.
- No true streaming audio inference yet.
- Mobile recording is scaffolded but not production-ready.
- Predictions are likely intent estimates, not translations.
- `stress_discomfort` is not a veterinary diagnosis. If behavior is unusual or persistent, contact a veterinarian.

## Next Steps

- Collect real clips per dog with consistent labels and context.
- Train and evaluate dog-specific models.
- Add mobile audio capture and upload.
- Add collar IMU and phone camera posture signals.
- Build real-time WebSocket inference after clip-based inference is useful.
