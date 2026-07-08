import argparse
import csv
import re
from pathlib import Path


SUPPORTED_AUDIO = {".wav", ".m4a", ".mp3", ".flac", ".ogg"}
FILENAME_PATTERN = re.compile(
    r"^(?P<recording_id>.+?)_(?P<breed>.+)_(?P<sex>M|F)_(?P<dog_prefix>dog)_(?P<dog_number>\d+)$",
    re.IGNORECASE,
)
DOG_FOLDER_PATTERN = re.compile(r"^dog_\d+$", re.IGNORECASE)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_audio_filename(audio_path: Path) -> dict[str, str]:
    match = FILENAME_PATTERN.match(audio_path.stem)
    folder_name = audio_path.parent.name

    if not match:
        return {
            "dog_id": folder_name if DOG_FOLDER_PATTERN.match(folder_name) else "",
            "audio_file_name": audio_path.name,
            "folder_name": folder_name,
            "audio_file_path": str(audio_path),
            "breed": "",
            "sex": "",
        }

    dog_id = f"{match.group('dog_prefix').lower()}_{match.group('dog_number')}"
    return {
        "dog_id": dog_id,
        "audio_file_name": audio_path.name,
        "folder_name": folder_name,
        "audio_file_path": str(audio_path),
        "breed": match.group("breed").lower(),
        "sex": match.group("sex").upper(),
    }


def find_audio_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*") if path.suffix.lower() in SUPPORTED_AUDIO)


def build_manifest(input_dir: Path, output_csv: Path) -> list[dict[str, str]]:
    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    rows = [parse_audio_filename(path) for path in find_audio_files(input_dir)]
    if not rows:
        raise SystemExit(f"No supported audio files found in {input_dir}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = ["dog_id", "audio_file_name", "folder_name", "audio_file_path", "breed", "sex"]
    with output_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Create a dog audio metadata CSV from dog_N folders.")
    parser.add_argument("--input-dir", type=Path, default=root / "backend" / "data" / "raw")
    parser.add_argument("--output-csv", type=Path, default=root / "backend" / "data" / "processed" / "dog_audio_manifest.csv")
    args = parser.parse_args()

    rows = build_manifest(args.input_dir, args.output_csv)
    print(f"Wrote {len(rows)} audio rows to {args.output_csv}")


if __name__ == "__main__":
    main()
