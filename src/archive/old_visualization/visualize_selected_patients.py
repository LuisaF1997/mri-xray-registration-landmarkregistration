from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk


LANDMARK_CSV = Path("data/processed/landmark_indices_clean.csv")
EXTRACTED_DIR = Path("data/extracted")
OUTPUT_DIR = Path("outputs/landmark_visualization/selected_patients")

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
    text = str(volume_file).lower()

    for c in candidates:
        if c.name.lower() == text:
            return c

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


def safe_index(idx, max_idx):
    return max(0, min(int(idx), max_idx))


def draw_points(ax, rows, view):
    for _, p in rows.iterrows():
        x = int(p["index_x"])
        y = int(p["index_y"])
        z = int(p["index_z"])
        idx = int(p["point_index"])

        if view in ["acquired", "xray"]:
            px, py = x, y
        elif view == "coronal":
            px, py = x, z
        elif view == "axial":
            px, py = y, z
        else:
            continue

        color = COLORS[idx % len(COLORS)]

        ax.scatter(px, py, s=90, color=color, edgecolors="black", linewidths=0.8)
        ax.text(
            px + 5,
            py + 5,
            f"{idx + 1}",
            color="yellow",
            fontsize=12,
            fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.5, pad=1),
        )


def plot_mri_patient(fig_axes, patient_id, rows):
    volume_file = rows.iloc[0]["volume_file"]
    image_path = find_image_path(patient_id, volume_file)

    image = sitk.ReadImage(str(image_path))
    volume = sitk.GetArrayFromImage(image).astype(np.float32)

    z_med = safe_index(rows["index_z"].median(), volume.shape[0] - 1)
    y_med = safe_index(rows["index_y"].median(), volume.shape[1] - 1)
    x_med = safe_index(rows["index_x"].median(), volume.shape[2] - 1)

    views = [
        ("MRI acquired/sagittal slice", volume[z_med, :, :], "acquired"),
        ("MRI coronal slice", volume[:, y_med, :], "coronal"),
        ("MRI axial slice", volume[:, :, x_med], "axial"),
    ]

    for ax, (title, img, view_key) in zip(fig_axes, views):
        ax.imshow(normalize(img), cmap="gray")
        draw_points(ax, rows, view_key)
        ax.set_title(title, fontsize=11)
        ax.axis("off")


def plot_xray_patient(ax, patient_id, rows):
    volume_file = rows.iloc[0]["volume_file"]
    image_path = find_image_path(patient_id, volume_file)

    image = sitk.ReadImage(str(image_path))
    volume = sitk.GetArrayFromImage(image).astype(np.float32)

    z_med = safe_index(rows["index_z"].median(), volume.shape[0] - 1)

    img = volume[z_med, :, :]
    ax.imshow(normalize(img), cmap="gray")
    draw_points(ax, rows, "xray")
    ax.set_title("X-ray with landmarks", fontsize=11)
    ax.axis("off")


def visualize_patient(df, patient_id):
    patient_df = df[df["patient_id"].astype(str) == str(patient_id)]

    if len(patient_df) == 0:
        print(f"Patient {patient_id}: no data found")
        return

    mri_rows = patient_df[patient_df["modality"] == "mri"]
    xray_rows = patient_df[patient_df["modality"] == "xray"]

    if len(mri_rows) == 0:
        print(f"Patient {patient_id}: no MRI landmarks found")
        return

    if len(xray_rows) == 0:
        print(f"Patient {patient_id}: no X-ray landmarks found")
        return

    fig = plt.figure(figsize=(16, 10))

    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.25])

    ax_mri_1 = fig.add_subplot(gs[0, 0])
    ax_mri_2 = fig.add_subplot(gs[0, 1])
    ax_mri_3 = fig.add_subplot(gs[0, 2])

    ax_xray = fig.add_subplot(gs[1, :])

    plot_mri_patient(
        [ax_mri_1, ax_mri_2, ax_mri_3],
        patient_id,
        mri_rows,
    )

    plot_xray_patient(ax_xray, patient_id, xray_rows)

    fig.suptitle(
        f"Patient {patient_id} - Ground Truth Landmarks",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = OUTPUT_DIR / f"patient_{patient_id}_landmarks.png"
    plt.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Saved: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(LANDMARK_CSV)
    df = df[df["inside_image"] == True]
    df = df[df["modality"].isin(["mri", "xray"])]

    available = set(df["patient_id"].astype(str).unique())

    print("Available patients:", len(available))

    for patient_id in SELECTED_PATIENTS:
        if patient_id not in available:
            print(f"Patient {patient_id} missing in landmark CSV")
            continue

        visualize_patient(df, patient_id)


if __name__ == "__main__":
    main()