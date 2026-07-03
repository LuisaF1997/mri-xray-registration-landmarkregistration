from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk


LANDMARK_CSV = Path("data/processed/landmark_indices_clean.csv")
EXTRACTED_DIR = Path("data/extracted")
OUTPUT_DIR = Path("outputs/landmark_visualization")
OUTPUT_PATH = OUTPUT_DIR / "landmark_grid_9_patients_multiview.png"

SELECTED_PATIENTS = ["16", "526", "788"]

COLORS = ["red", "lime", "dodgerblue", "orange", "magenta", "cyan"]


def normalize(img):
    p1, p99 = np.percentile(img, [1, 99])
    img = np.clip(img, p1, p99)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def find_image_path(patient_id, volume_file):
    patient_dir = EXTRACTED_DIR / str(patient_id)
    exact = patient_dir / str(volume_file)

    if exact.exists():
        return exact

    candidates = list(patient_dir.glob("*.nrrd"))

    for c in candidates:
        if c.name.lower() == str(volume_file).lower():
            return c

    text = str(volume_file).lower()

    for c in candidates:
        cname = c.name.lower()

        if "lateral" in text and "lateral" in cname:
            return c
        if "lat" in text and "lat" in cname:
            return c
        if "ap" in text and "ap" in cname:
            return c
        if "pd" in text and "pd" in cname:
            return c
        if "sag" in text and "sag" in cname:
            return c
        if "fse" in text and "fse" in cname:
            return c
        if "spair" in text and "spair" in cname:
            return c
        if "clear" in text and "clear" in cname:
            return c

    raise FileNotFoundError(f"Image not found for patient {patient_id}: {volume_file}")


def safe_slice(v, idx, axis):
    if axis == 0:
        idx = max(0, min(int(idx), v.shape[0] - 1))
        return v[idx, :, :]
    if axis == 1:
        idx = max(0, min(int(idx), v.shape[1] - 1))
        return v[:, idx, :]
    if axis == 2:
        idx = max(0, min(int(idx), v.shape[2] - 1))
        return v[:, :, idx]
    raise ValueError("axis must be 0, 1, or 2")


def draw_points(ax, rows, view):
    """
    view:
        sagittal_original -> image = volume[z, :, :]  use x/y
        axial             -> image = volume[z, :, :]  use x/y
        coronal           -> image = volume[:, y, :]  use x/z
        sagittal          -> image = volume[:, :, x]  use y/z
        xray              -> image = volume[z, :, :]  use x/y
    """

    for _, p in rows.iterrows():
        x = int(p["index_x"])
        y = int(p["index_y"])
        z = int(p["index_z"])
        idx = int(p["point_index"])

        if view in ["sagittal_original", "axial", "xray"]:
            px, py = x, y
        elif view == "coronal":
            px, py = x, z
        elif view == "sagittal":
            px, py = y, z
        else:
            continue

        color = COLORS[idx % len(COLORS)]
        ax.scatter(px, py, s=35, color=color)
        ax.text(px + 3, py + 3, str(idx + 1), color="yellow", fontsize=8)


def plot_mri_views(axes, patient_id, rows):
    if len(rows) == 0:
        for ax in axes:
            ax.axis("off")
        axes[0].set_title("MRI missing")
        return

    volume_file = rows.iloc[0]["volume_file"]
    image_path = find_image_path(patient_id, volume_file)

    image = sitk.ReadImage(str(image_path))
    volume = sitk.GetArrayFromImage(image).astype(np.float32)

    z_med = int(rows["index_z"].median())
    y_med = int(rows["index_y"].median())
    x_med = int(rows["index_x"].median())

    # Most MRI datasets are sagittal acquisitions.
    # Therefore volume[z,:,:] is usually the clinically useful sagittal slice.
    views = [
        ("MRI sagittal/acquired", safe_slice(volume, z_med, axis=0), "sagittal_original"),
        ("MRI coronal", safe_slice(volume, y_med, axis=1), "coronal"),
        ("MRI axial", safe_slice(volume, x_med, axis=2), "sagittal"),
    ]

    for ax, (title, img, view_key) in zip(axes, views):
        ax.imshow(normalize(img), cmap="gray")
        draw_points(ax, rows, view_key)
        ax.set_title(title, fontsize=8)
        ax.axis("off")


def plot_xray_view(ax, patient_id, rows):
    if len(rows) == 0:
        ax.axis("off")
        ax.set_title("X-ray missing")
        return

    volume_file = rows.iloc[0]["volume_file"]
    image_path = find_image_path(patient_id, volume_file)

    image = sitk.ReadImage(str(image_path))
    volume = sitk.GetArrayFromImage(image).astype(np.float32)

    z_med = int(rows["index_z"].median())
    img = safe_slice(volume, z_med, axis=0)

    ax.imshow(normalize(img), cmap="gray")
    draw_points(ax, rows, "xray")
    ax.set_title("X-ray", fontsize=8)
    ax.axis("off")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(LANDMARK_CSV)
    df = df[df["inside_image"] == True]
    df = df[df["modality"].isin(["mri", "xray"])]

    patient_ids = set(df["patient_id"].astype(str).unique())

selected = [
    pid for pid in SELECTED_PATIENTS
    if pid in patient_ids
]

missing = [
    pid for pid in SELECTED_PATIENTS
    if pid not in patient_ids
]

if missing:
    print("Missing patients:")
    print(", ".join(missing))
    # 4 columns per patient row:
    # MRI sagittal/acquired | MRI coronal | MRI axial | X-ray
    fig, axes = plt.subplots(
        len(selected),
        4,
        figsize=(14, len(selected) * 3.2),
    )

    if len(selected) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, patient_id in enumerate(selected):
        patient_df = df[df["patient_id"].astype(str) == patient_id]

        mri_rows = patient_df[patient_df["modality"] == "mri"]
        xray_rows = patient_df[patient_df["modality"] == "xray"]

        plot_mri_views(
            axes[row_idx, 0:3],
            patient_id,
            mri_rows,
        )

        plot_xray_view(
            axes[row_idx, 3],
            patient_id,
            xray_rows,
        )

        axes[row_idx, 0].text(
            -0.12,
            0.5,
            f"Patient {patient_id}",
            transform=axes[row_idx, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=10,
            fontweight="bold",
        )

    plt.suptitle(
        "Landmark quality check: MRI multiview + X-ray",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(OUTPUT_PATH, dpi=200)
    plt.close()

    print(f"Saved landmark grid to: {OUTPUT_PATH}")
    print("Selected patients:")
    print(", ".join(selected))


if __name__ == "__main__":
    main()