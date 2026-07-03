from pathlib import Path
import json
import pandas as pd


EXTRACTED_DIR = Path("data/extracted")
OUTPUT_DIR = Path("data/processed")
OUTPUT_CSV = OUTPUT_DIR / "landmarks.csv"


def load_mrk_points(mrk_file: Path):
    with open(mrk_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    points = []

    for markup in data.get("markups", []):
        for idx, cp in enumerate(markup.get("controlPoints", [])):
            position = cp.get("position", [None, None, None])

            points.append({
                "point_index": idx,
                "label": cp.get("label", ""),
                "x": position[0],
                "y": position[1],
                "z": position[2],
            })

    return points


def classify_landmark_file(file_name: str):
    name = file_name.lower()

    if "from" in name:
        return "from_mri"

    if "to" in name:
        return "to_xray"

    return "unknown"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    patient_folders = sorted([p for p in EXTRACTED_DIR.iterdir() if p.is_dir()])

    print(f"Found {len(patient_folders)} extracted patient folders.")

    for patient_folder in patient_folders:
        patient_id = patient_folder.name

        mrk_files = sorted(patient_folder.glob("*.mrk.json"))

        if len(mrk_files) == 0:
            print(f"Warning: no landmark files found for patient {patient_id}")
            continue

        for mrk_file in mrk_files:
            landmark_type = classify_landmark_file(mrk_file.name)
            points = load_mrk_points(mrk_file)

            for p in points:
                rows.append({
                    "patient_id": patient_id,
                    "file_name": mrk_file.name,
                    "landmark_type": landmark_type,
                    "point_index": p["point_index"],
                    "label": p["label"],
                    "x": p["x"],
                    "y": p["y"],
                    "z": p["z"],
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n=============================")
    print("LANDMARK DATASET SUMMARY")
    print("=============================")
    print(f"Patients: {df['patient_id'].nunique()}")
    print(f"Total landmark rows: {len(df)}")
    print(f"Saved to: {OUTPUT_CSV}")

    print("\nLandmark type counts:")
    print(df["landmark_type"].value_counts())

    print("\nPoints per patient/type:")
    print(
        df.groupby(["patient_id", "landmark_type"])
        .size()
        .reset_index(name="num_points")
        .head(20)
    )


if __name__ == "__main__":
    main()