import argparse
from pathlib import Path

import pandas as pd


INTENT_LABELS = ["outside_bathroom", "food_water", "play", "attention", "stress_discomfort", "unknown"]
LOCATIONS = ["door", "kitchen", "couch", "crate", "outside", "unknown"]
SITUATIONS = ["before_walk", "before_food", "owner_leaving", "stranger_nearby", "toy_visible", "unknown"]


def generate_dummy_metadata(output_csv: Path, rows: int) -> pd.DataFrame:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for index in range(rows):
        label = INTENT_LABELS[index % len(INTENT_LABELS)]
        data.append(
            {
                "dog_id": 1,
                "file_path": f"data/raw/replace_with_real_clip_{index + 1}.wav",
                "location_context": LOCATIONS[index % len(LOCATIONS)],
                "situation_context": SITUATIONS[index % len(SITUATIONS)],
                "owner_label": label,
                "outcome_label": label,
                "notes": "Dummy metadata only. Replace file_path with a real dog audio clip.",
            }
        )
    frame = pd.DataFrame(data)
    frame.to_csv(output_csv, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fake metadata rows for DogBridge development.")
    parser.add_argument("--output-csv", type=Path, default=Path("data/processed/dummy_metadata.csv"))
    parser.add_argument("--rows", type=int, default=12)
    args = parser.parse_args()

    frame = generate_dummy_metadata(args.output_csv, args.rows)
    print(f"Wrote {len(frame)} dummy metadata rows to {args.output_csv}")
    print("No fake audio was generated. Replace file_path values with real dog recordings before training.")


if __name__ == "__main__":
    main()

