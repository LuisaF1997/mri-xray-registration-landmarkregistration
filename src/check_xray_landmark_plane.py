from pathlib import Path
import pandas as pd
import numpy as np


LANDMARK_CSV = Path("data/processed/landmark_indices_clean.csv")
OUTPUT_CSV = Path("data/processed/xray_landmark_plane_check.csv")


def main():
    df = pd.read_csv(LANDMARK_CSV)

    xray = df[
        (df["modality"] == "xray")
        & (df["inside_image"] == True)
    ].copy()

    rows = []

    for patient_id, group in xray.groupby("patient_id"):
        pts = group[["physical_x", "physical_y", "physical_z"]].values.astype(float)

        std_x = np.std(pts[:, 0])
        std_y = np.std(pts[:, 1])
        std_z = np.std(pts[:, 2])

        range_x = np.ptp(pts[:, 0])
        range_y = np.ptp(pts[:, 1])
        range_z = np.ptp(pts[:, 2])

        rows.append({
            "patient_id": patient_id,
            "num_points": len(group),
            "std_x": std_x,
            "std_y": std_y,
            "std_z": std_z,
            "range_x": range_x,
            "range_y": range_y,
            "range_z": range_z,
            "likely_plane_axis": min(
                [("x", range_x), ("y", range_y), ("z", range_z)],
                key=lambda x: x[1],
            )[0],
            "smallest_range_mm": min(range_x, range_y, range_z),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False)

    print("\n=============================")
    print("XRAY LANDMARK PLANE CHECK")
    print("=============================")
    print(f"Patients checked: {len(out)}")
    print(f"Saved to: {OUTPUT_CSV}")

    print("\nLikely plane axis counts:")
    print(out["likely_plane_axis"].value_counts())

    print("\nSmallest ranges:")
    print(out[["patient_id", "num_points", "likely_plane_axis", "smallest_range_mm"]].head(20))


if __name__ == "__main__":
    main()