from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import SimpleITK as sitk


LANDMARK_CSV = Path("data/processed/landmark_indices_clean.csv")
EXTRACTED_DIR = Path("data/extracted")
OUTPUT_DIR = Path("outputs/landmark_visualization")
OUTPUT_PDF = OUTPUT_DIR / "all_patients_landmarks.pdf"

COLORS = ["red", "lime", "dodgerblue", "orange", "magenta", "cyan"]


def normalize(img):
    p1, p99 = np.percentile(img, [1, 99])
    img = np.clip(img, p1, p99)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def safe_index(value, max_index):
    return max(0, min(int(value), max_index))


def sort_patient_id(pid):
    pid = str(pid)
    if pid.isdigit():
        return (0, int(pid))
    return (1, pid)


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

    keywords = [
        "lateral", "lat", "ap", "xray",
        "pd", "sag", "fse", "spair", "clear", "smffe", "tse"
    ]

    best_file = None
    best_score = -1

    for c in candidates:
        cname = c.name.lower()
        score = sum(1 for k in keywords if k in text and k in cname)

        if score > best_score:
            best_score = score
            best_file = c

    if best_file is not None and best_score > 0:
        return best_file

    raise FileNotFoundError(f"Image not found for patient {patient_id}: {volume_file}")


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
            str(idx + 1),
            color="yellow",
            fontsize=11,
            fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.55, pad=1),
        )


def plot_mri_views(axes, patient_id, rows):
    volume_file = rows.iloc[0]["volume_file"]
    image_path = find_image_path(patient_id, volume_file)

    image = sitk.ReadImage(str(image_path))
    volume = sitk.GetArrayFromImage(image).astype(np.float32)

    z_med = safe_index(rows["index_z"].median(), volume.shape[0] - 1)
    y_med = safe_index(rows["index_y"].median(), volume.shape[1] - 1)
    x_med = safe_index(rows["index_x"].median(), volume.shape[2] - 1)

    views = [
        ("MRI acquired/sagittal", volume[z_med, :, :], "acquired"),
        ("MRI coronal", volume[:, y_med, :], "coronal"),
        ("MRI axial", volume[:, :, x_med], "axial"),
    ]

    for ax, (title, img, view_key) in zip(axes, views):
        ax.imshow(normalize(img), cmap="gray")
        draw_points(ax, rows, view_key)
        ax.set_title(title, fontsize=11)
        ax.axis("off")


def plot_xray(ax, patient_id, rows):
    volume_file = rows.iloc[0]["volume_file"]
    image_path = find_image_path(patient_id, volume_file)

    image = sitk.ReadImage(str(image_path))
    volume = sitk.GetArrayFromImage(image).astype(np.float32)

    z_med = safe_index(rows["index_z"].median(), volume.shape[0] - 1)
    img = volume[z_med, :, :]

    ax.imshow(normalize(img), cmap="gray")
    draw_points(ax, rows, "xray")
    ax.set_title("X-ray", fontsize=11)
    ax.axis("off")


def add_coordinate_table(ax, mri_rows, xray_rows):
    ax.axis("off")

    table_rows = []

    for i in sorted(set(mri_rows["point_index"]).union(set(xray_rows["point_index"]))):
        mri = mri_rows[mri_rows["point_index"] == i]
        xray = xray_rows[xray_rows["point_index"] == i]

        if len(mri) > 0:
            m = mri.iloc[0]
            mri_text = f"({int(m['index_x'])}, {int(m['index_y'])}, {int(m['index_z'])})"
        else:
            mri_text = "-"

        if len(xray) > 0:
            x = xray.iloc[0]
            xray_text = f"({int(x['index_x'])}, {int(x['index_y'])}, {int(x['index_z'])})"
        else:
            xray_text = "-"

        table_rows.append([str(int(i) + 1), mri_text, xray_text])

    table = ax.table(
        cellText=table_rows,
        colLabels=["Point", "MRI index (x,y,z)", "X-ray index (x,y,z)"],
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    ax.set_title("Landmark index coordinates", fontsize=11)


def create_patient_figure(df, patient_id):
    patient_df = df[df["patient_id"].astype(str) == str(patient_id)]

    mri_rows = patient_df[patient_df["modality"] == "mri"]
    xray_rows = patient_df[patient_df["modality"] == "xray"]

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1.25, 0.6])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :])
    ax5 = fig.add_subplot(gs[2, :])

    if len(mri_rows) > 0:
        plot_mri_views([ax1, ax2, ax3], patient_id, mri_rows)
    else:
        for ax in [ax1, ax2, ax3]:
            ax.axis("off")
        ax1.set_title("MRI missing")

    if len(xray_rows) > 0:
        plot_xray(ax4, patient_id, xray_rows)
    else:
        ax4.axis("off")
        ax4.set_title("X-ray missing")

    add_coordinate_table(ax5, mri_rows, xray_rows)

    fig.suptitle(
        f"Patient {patient_id} - Ground Truth Landmarks",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(LANDMARK_CSV)
    df = df[df["inside_image"] == True]
    df = df[df["modality"].isin(["mri", "xray"])]

    patient_ids = sorted(
        df["patient_id"].astype(str).unique(),
        key=sort_patient_id,
    )

    print(f"Patients to visualize: {len(patient_ids)}")
    print(f"Saving PDF to: {OUTPUT_PDF}")

    with PdfPages(OUTPUT_PDF) as pdf:
        for patient_id in patient_ids:
            try:
                print(f"Creating page for patient {patient_id}")
                fig = create_patient_figure(df, patient_id)
                pdf.savefig(fig)
                plt.close(fig)
            except Exception as e:
                print(f"Skipping patient {patient_id}: {e}")

    print("Done.")
    print(f"Saved: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()