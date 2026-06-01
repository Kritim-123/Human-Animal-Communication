import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd


SUPPORTED_AUDIO = {".wav", ".m4a", ".mp3", ".flac", ".ogg"}
TARGET_SAMPLE_RATE = 22050


def _stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
    }


def extract_feature_dict(audio_path: Path) -> dict[str, float]:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    y, sr = librosa.load(str(audio_path), sr=TARGET_SAMPLE_RATE, mono=True)
    if y.size == 0:
        raise ValueError(f"Audio file is empty or unreadable: {audio_path}")

    features: dict[str, float] = {}
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for index, row in enumerate(mfcc, start=1):
        features.update(_stats(f"mfcc_{index}", row))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.update(_stats("chroma", chroma))
    features.update(_stats("spectral_centroid", librosa.feature.spectral_centroid(y=y, sr=sr)))
    features.update(_stats("spectral_bandwidth", librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    features.update(_stats("zero_crossing_rate", librosa.feature.zero_crossing_rate(y)))
    features.update(_stats("rms", librosa.feature.rms(y=y)))
    return features


def extract_folder(input_dir: Path, output_csv: Path, metadata_csv: Path | None = None) -> pd.DataFrame:
    audio_files = sorted(path for path in input_dir.rglob("*") if path.suffix.lower() in SUPPORTED_AUDIO)
    if not audio_files:
        raise SystemExit(f"No supported audio files found in {input_dir}. Add real dog clips before extracting features.")

    metadata = None
    if metadata_csv:
        metadata = pd.read_csv(metadata_csv)
        if "file_path" not in metadata.columns:
            raise SystemExit("Metadata CSV must include a file_path column.")

    rows = []
    for audio_file in audio_files:
        try:
            row = {"file_path": str(audio_file), **extract_feature_dict(audio_file)}
            if metadata is not None:
                matches = metadata[metadata["file_path"].astype(str) == str(audio_file)]
                if not matches.empty:
                    for column, value in matches.iloc[0].items():
                        if column != "file_path":
                            row[column] = value
            rows.append(row)
        except Exception as exc:
            print(f"Skipping {audio_file}: {exc}")

    if not rows:
        raise SystemExit("No features could be extracted. Check that audio files are valid.")

    frame = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract DogBridge baseline audio features.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/processed/features.csv"))
    parser.add_argument("--metadata-csv", type=Path, default=None)
    args = parser.parse_args()

    frame = extract_folder(args.input_dir, args.output_csv, args.metadata_csv)
    print(f"Wrote {len(frame)} feature rows to {args.output_csv}")


if __name__ == "__main__":
    main()

