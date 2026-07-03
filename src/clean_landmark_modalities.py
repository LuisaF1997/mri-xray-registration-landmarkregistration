from pathlib import Path
import pandas as pd


INPUT_CSV = Path("data/processed/landmark_indices.csv")
OUTPUT_CSV = Path("data/processed/landmark_indices_clean.csv")


def infer_modality(row):
    current = str(row["modality"]).lower()

    if current in ["mri", "xray"]:
        return current

    text = " ".join([
        str(row.get("volume_name", "")),
        str(row.get("volume_file", "")),
        str(row.get("mrk_file", "")),
    ]).lower()

    xray_keywords = [
        "xray",
        "x-ray",
        "lateral",
        "lat",
        "ap",
        "kneelat",
        "knie lat",
        "knie ap",
        "r_1",
        "l_1",
    ]

    mri_keywords = [
        "mri",
        "mr",
        "pd",
        "tse",
        "fse",
        "sag",
        "spair",
        "clear",
        "smffe",
        "fs",
    ]

    xray_score = sum(k in text for k in xray_keywords)
    mri_score = sum(k in text for k in mri_keywords)

    if xray_score > mri_score:
        return "xray"

    if mri_score > xray_score:
        return "mri"

    return "unknown"


def main():
    df = pd.read_csv(INPUT_CSV)

    print("\nBefore:")
    print(df["modality"].value_counts())

    df["modality_original"] = df["modality"]
    df["modality"] = df.apply(infer_modality, axis=1)

    df.to_csv(OUTPUT_CSV, index=False)

    print("\nAfter:")
    print(df["modality"].value_counts())

    print(f"\nSaved cleaned CSV to: {OUTPUT_CSV}")

    unknown = df[df["modality"] == "unknown"]

    if len(unknown) > 0:
        print("\nStill unknown:")
        print(
            unknown[
                [
                    "patient_id",
                    "volume_name",
                    "volume_file",
                    "mrk_file",
                    "label",
                ]
            ]
        )


if __name__ == "__main__":
    main()