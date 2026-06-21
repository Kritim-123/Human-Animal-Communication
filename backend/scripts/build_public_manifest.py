import argparse
from pathlib import Path

import pandas as pd


PUBLIC_MANIFEST_COLUMNS = [
    "source_dataset",
    "source_url",
    "license",
    "file_path",
    "filename",
    "dog_id",
    "breed",
    "sex",
    "sound_label",
    "context_label",
    "intent_label",
    "usable_for_intent",
    "file_exists",
    "notes",
]


def build_dogspeak_manifest(input_metadata: Path, audio_root: Path, output_csv: Path) -> pd.DataFrame:
    if not input_metadata.exists():
        raise SystemExit(f"DogSpeak metadata not found: {input_metadata}")

    metadata = pd.read_csv(input_metadata)
    required = {"filename", "breed", "sex", "dog_id"}
    missing = required - set(metadata.columns)
    if missing:
        raise SystemExit(f"DogSpeak metadata is missing required columns: {sorted(missing)}")

    rows = []
    for _, item in metadata.iterrows():
        filename = str(item["filename"])
        dog_id = str(item["dog_id"])
        source_file_path = ""
        for column in ["file_path", "path", "audio", "audio_path"]:
            if column in metadata.columns and pd.notna(item[column]):
                source_file_path = str(item[column])
                break
        file_path = str(audio_root / source_file_path) if source_file_path else str(audio_root / dog_id / filename)
        rows.append(
            {
                "source_dataset": "DogSpeak",
                "source_url": "https://huggingface.co/datasets/ArlingtonCL2/DogSpeak_Dataset",
                "license": "CC BY-NC-SA 4.0",
                "file_path": file_path,
                "filename": filename,
                "dog_id": dog_id,
                "breed": item["breed"],
                "sex": item["sex"],
                "sound_label": "bark_sequence",
                "context_label": "",
                "intent_label": "",
                "usable_for_intent": False,
                "file_exists": False,
                "notes": "Public canine vocalization data. Use for acoustic representation learning, not DogBridge intent labels.",
            }
        )

    frame = pd.DataFrame(rows, columns=PUBLIC_MANIFEST_COLUMNS)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a DogBridge public audio manifest.")
    parser.add_argument("--dataset", choices=["dogspeak"], required=True)
    parser.add_argument("--input-metadata", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=Path("data/public/manifests/dogspeak_manifest.csv"))
    args = parser.parse_args()

    if args.dataset == "dogspeak":
        frame = build_dogspeak_manifest(args.input_metadata, args.audio_root, args.output_csv)
    else:
        raise SystemExit(f"Unsupported dataset: {args.dataset}")

    print(f"Wrote {len(frame)} public audio rows to {args.output_csv}")
    print("File existence is not checked here to avoid slow scans on mounted drives.")


if __name__ == "__main__":
    main()
