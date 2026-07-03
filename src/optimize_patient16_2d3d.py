from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


LANDMARK_CSV = Path("data/processed/landmark_indices_clean.csv")
EXTRACTED_DIR = Path("data/extracted")
OUTPUT_DIR = Path("outputs/landmark_registration")
OUTPUT_CSV = OUTPUT_DIR / "optimized_2d3d_all_patients.csv"


def load_gt_matrix(patient_id):
    mrml_file = next((EXTRACTED_DIR / str(patient_id)).glob("*.mrml"))

    tree = ET.parse(mrml_file)
    root = tree.getroot()

    for node in root.iter():
        attrs = node.attrib

        if attrs.get("name") == "XrayMRI_Registration_Transform":
            values = [float(v) for v in attrs["matrixTransformToParent"].split()]
            return np.array(values, dtype=np.float64).reshape(4, 4)

    raise ValueError(f"Ground truth transform not found for patient {patient_id}")


def lps_to_ras(points):
    pts = points.copy()
    pts[:, 0] *= -1
    pts[:, 1] *= -1
    return pts


def ras_to_lps(points):
    return lps_to_ras(points)


def make_matrix(params):
    rx, ry, rz, tx, ty, tz = params

    M = np.eye(4)
    M[:3, :3] = R.from_rotvec([rx, ry, rz]).as_matrix()
    M[:3, 3] = [tx, ty, tz]

    return M


def matrix_to_params(M):
    rotvec = R.from_matrix(M[:3, :3]).as_rotvec()
    trans = M[:3, 3]
    return np.concatenate([rotvec, trans])


def transform_points(points, M):
    pts_h = np.hstack([points, np.ones((len(points), 1))])
    out = (M @ pts_h.T).T
    return out[:, :3]


def residuals(params, mri_pts_ras, xray_pts_lps):
    M = make_matrix(params)

    pred_ras = transform_points(mri_pts_ras, M)
    pred_lps = ras_to_lps(pred_ras)

    # X-ray points lie on the z=0 plane.
    # We compare only the in-plane x/y coordinates.
    return (pred_lps[:, :2] - xray_pts_lps[:, :2]).reshape(-1)


def rmse_2d(a, b):
    return float(np.sqrt(np.mean(np.sum((a[:, :2] - b[:, :2]) ** 2, axis=1))))


def evaluate_patient(patient_id, patient_df):
    mri = patient_df[patient_df["modality"] == "mri"].sort_values("point_index")
    xray = patient_df[patient_df["modality"] == "xray"].sort_values("point_index")

    if len(mri) < 3 or len(xray) < 3:
        raise ValueError("Fewer than 3 MRI or X-ray landmarks")

    mri_pts_lps = mri[["physical_x", "physical_y", "physical_z"]].values.astype(float)
    xray_pts_lps = xray[["physical_x", "physical_y", "physical_z"]].values.astype(float)

    mri_pts_ras = lps_to_ras(mri_pts_lps)

    gt_xray_to_mri_ras = load_gt_matrix(patient_id)
    gt_mri_to_xray_ras = np.linalg.inv(gt_xray_to_mri_ras)

    best = None

    for perm in itertools.permutations(range(len(xray_pts_lps)), 3):
        xray_perm = xray_pts_lps[list(perm)]

        # Validation mode:
        # Start close to ground truth to test whether optimization is mathematically stable.
        x0 = matrix_to_params(gt_mri_to_xray_ras)

        result = least_squares(
            residuals,
            x0,
            args=(mri_pts_ras, xray_perm),
            max_nfev=5000,
        )

        M_opt = make_matrix(result.x)

        pred_ras = transform_points(mri_pts_ras, M_opt)
        pred_lps = ras_to_lps(pred_ras)

        error = rmse_2d(pred_lps, xray_perm)
        matrix_diff = float(np.linalg.norm(M_opt - gt_mri_to_xray_ras))

        row = {
            "patient_id": patient_id,
            "num_mri_points": len(mri),
            "num_xray_points": len(xray),
            "permutation": str(list(perm)),
            "rmse_2d_mm": error,
            "matrix_frobenius_diff_to_gt": matrix_diff,
            "optimizer_success": result.success,
            "optimizer_message": result.message,
        }

        for i in range(4):
            for j in range(4):
                row[f"opt_m{i}{j}"] = M_opt[i, j]
                row[f"gt_m{i}{j}"] = gt_mri_to_xray_ras[i, j]

        if best is None or error < best["rmse_2d_mm"]:
            best = row

    return best


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(LANDMARK_CSV)
    df = df[df["inside_image"] == True]
    df = df[df["modality"].isin(["mri", "xray"])]

    rows = []

    for patient_id, patient_df in df.groupby("patient_id"):
        try:
            result = evaluate_patient(patient_id, patient_df)
            rows.append(result)

            print(
                f"Patient {patient_id}: "
                f"RMSE={result['rmse_2d_mm']:.4f} mm | "
                f"Matrix diff={result['matrix_frobenius_diff_to_gt']:.4f}"
            )

        except Exception as e:
            print(f"Patient {patient_id}: ERROR {e}")

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False)

    print("\n=============================")
    print("2D/3D OPTIMIZATION ALL PATIENTS")
    print("=============================")
    print(f"Patients evaluated: {len(out)}")
    print(f"Saved to: {OUTPUT_CSV}")

    print("\n2D reprojection RMSE:")
    print(f"Mean:   {out['rmse_2d_mm'].mean():.4f} mm")
    print(f"Median: {out['rmse_2d_mm'].median():.4f} mm")
    print(f"Max:    {out['rmse_2d_mm'].max():.4f} mm")

    print("\nMatrix difference to ground truth:")
    print(f"Mean:   {out['matrix_frobenius_diff_to_gt'].mean():.4f}")
    print(f"Median: {out['matrix_frobenius_diff_to_gt'].median():.4f}")
    print(f"Max:    {out['matrix_frobenius_diff_to_gt'].max():.4f}")

    print("\nWorst patients:")
    print(
        out[
            [
                "patient_id",
                "rmse_2d_mm",
                "matrix_frobenius_diff_to_gt",
                "permutation",
            ]
        ]
        .sort_values("rmse_2d_mm", ascending=False)
        .head(15)
    )


if __name__ == "__main__":
    main()