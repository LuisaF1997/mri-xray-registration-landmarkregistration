from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET


LANDMARK_CSV = Path("data/processed/landmark_indices_clean.csv")
EXTRACTED_DIR = Path("data/extracted")
OUTPUT_DIR = Path("outputs/landmark_registration")
OUTPUT_CSV = OUTPUT_DIR / "2d3d_landmark_permutation_results.csv"


def load_transform_from_mrml(patient_id):
    patient_dir = EXTRACTED_DIR / str(patient_id)
    mrml_file = next(patient_dir.glob("*.mrml"))

    tree = ET.parse(mrml_file)
    root = tree.getroot()

    for node in root.iter():
        attrs = node.attrib

        if attrs.get("name") == "XrayMRI_Registration_Transform":
            matrix_string = attrs.get("matrixTransformToParent")

            if matrix_string is None:
                raise ValueError(f"No matrixTransformToParent for patient {patient_id}")

            values = [float(v) for v in matrix_string.split()]
            matrix = np.array(values, dtype=np.float64).reshape(4, 4)

            return matrix

    raise ValueError(f"No XrayMRI_Registration_Transform found for patient {patient_id}")


def to_homogeneous(points):
    ones = np.ones((points.shape[0], 1))
    return np.hstack([points, ones])


def from_homogeneous(points_h):
    return points_h[:, :3] / points_h[:, 3:4]


def lps_to_ras(points):
    out = points.copy()
    out[:, 0] *= -1
    out[:, 1] *= -1
    return out


def ras_to_lps(points):
    return lps_to_ras(points)


def transform_points(points, matrix):
    points_h = to_homogeneous(points)
    transformed = (matrix @ points_h.T).T
    return from_homogeneous(transformed)


def rmse(a, b):
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def get_points(patient_df, modality):
    rows = patient_df[patient_df["modality"] == modality].copy()
    rows = rows.sort_values("point_index")

    points = rows[["physical_x", "physical_y", "physical_z"]].values.astype(np.float64)
    indices = rows["point_index"].values.astype(int)

    return points, indices


def evaluate_patient(patient_id, patient_df):
    xray_points_lps, xray_indices = get_points(patient_df, "xray")
    mri_points_lps, mri_indices = get_points(patient_df, "mri")

    if len(xray_points_lps) < 3 or len(mri_points_lps) < 3:
        return []

    T_xray_to_mri_ras = load_transform_from_mrml(patient_id)
    T_mri_to_xray_ras = np.linalg.inv(T_xray_to_mri_ras)

    results = []

    mri_combos = list(itertools.combinations(range(len(mri_points_lps)), 3))
    xray_combos = list(itertools.combinations(range(len(xray_points_lps)), 3))

    for mri_combo in mri_combos:
        mri_subset_lps = mri_points_lps[list(mri_combo)]
        mri_subset_indices = mri_indices[list(mri_combo)]

        for xray_combo in xray_combos:
            xray_subset_lps = xray_points_lps[list(xray_combo)]
            xray_subset_indices = xray_indices[list(xray_combo)]

            for perm in itertools.permutations(range(3)):
                xray_perm_lps = xray_subset_lps[list(perm)]
                xray_perm_indices = xray_subset_indices[list(perm)]

                # Mode 1: use points directly as stored
                pred_xray_direct = transform_points(mri_subset_lps, T_mri_to_xray_ras)
                error_direct_2d = rmse(pred_xray_direct[:, :2], xray_perm_lps[:, :2])

                # Mode 2: landmarks are LPS, transform matrix is RAS
                mri_ras = lps_to_ras(mri_subset_lps)
                pred_xray_ras = transform_points(mri_ras, T_mri_to_xray_ras)
                pred_xray_lps = ras_to_lps(pred_xray_ras)
                error_lps_ras_2d = rmse(pred_xray_lps[:, :2], xray_perm_lps[:, :2])

                results.append({
                    "patient_id": patient_id,
                    "mri_point_indices": str(list(mri_subset_indices)),
                    "xray_point_indices": str(list(xray_perm_indices)),
                    "xray_combo_original_indices": str(list(xray_subset_indices)),
                    "permutation": str(list(perm)),
                    "error_direct_2d_mm": error_direct_2d,
                    "error_lps_ras_2d_mm": error_lps_ras_2d,
                    "best_mode": "direct" if error_direct_2d < error_lps_ras_2d else "lps_ras",
                    "best_error_2d_mm": min(error_direct_2d, error_lps_ras_2d),
                })

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(LANDMARK_CSV)
    df = df[df["inside_image"] == True]
    df = df[df["modality"].isin(["mri", "xray"])]

    all_results = []

    for patient_id, patient_df in df.groupby("patient_id"):
        try:
            results = evaluate_patient(patient_id, patient_df)
            all_results.extend(results)
        except Exception as e:
            print(f"Patient {patient_id}: ERROR {e}")

    result_df = pd.DataFrame(all_results)

    best_df = (
        result_df
        .sort_values("best_error_2d_mm")
        .groupby("patient_id")
        .head(1)
        .reset_index(drop=True)
    )

    result_df.to_csv(OUTPUT_CSV, index=False)
    best_df.to_csv(OUTPUT_DIR / "best_2d3d_landmark_results.csv", index=False)

    print("\n=============================")
    print("2D/3D LANDMARK REGISTRATION TEST")
    print("=============================")
    print(f"Patients evaluated: {best_df['patient_id'].nunique()}")
    print(f"All permutations saved to: {OUTPUT_CSV}")
    print(f"Best results saved to: {OUTPUT_DIR / 'best_2d3d_landmark_results.csv'}")

    print("\nBest 2D reprojection error:")
    print(f"Mean:   {best_df['best_error_2d_mm'].mean():.4f} mm")
    print(f"Median: {best_df['best_error_2d_mm'].median():.4f} mm")
    print(f"Max:    {best_df['best_error_2d_mm'].max():.4f} mm")

    print("\nBest coordinate mode counts:")
    print(best_df["best_mode"].value_counts())

    print("\nWorst patients:")
    print(
        best_df[
            [
                "patient_id",
                "best_error_2d_mm",
                "best_mode",
                "mri_point_indices",
                "xray_point_indices",
            ]
        ]
        .sort_values("best_error_2d_mm", ascending=False)
        .head(15)
    )


if __name__ == "__main__":
    main()