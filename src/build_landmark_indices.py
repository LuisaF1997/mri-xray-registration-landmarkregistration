from pathlib import Path
import json
import xml.etree.ElementTree as ET
import urllib.parse

import numpy as np
import pandas as pd
import SimpleITK as sitk


EXTRACTED_DIR = Path("data/extracted")
OUTPUT_DIR = Path("data/processed")
OUTPUT_CSV = OUTPUT_DIR / "landmark_indices.csv"


def load_points(mrk_file):
    with open(mrk_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    points = []
    for markup in data.get("markups", []):
        for idx, cp in enumerate(markup.get("controlPoints", [])):
            points.append({
                "point_index": idx,
                "label": cp.get("label", ""),
                "position": cp.get("position", []),
                "associatedNodeID": cp.get("associatedNodeID", ""),
            })
    return points


def parse_references(ref_string):
    refs = {}
    for part in ref_string.split(";"):
        if ":" in part:
            key, value = part.split(":", 1)
            refs[key] = value
    return refs


def parse_mrml(patient_dir):
    mrml_file = next(patient_dir.glob("*.mrml"))
    tree = ET.parse(mrml_file)
    root = tree.getroot()

    volume_nodes = {}
    storage_nodes = {}

    for node in root.iter():
        attrs = node.attrib
        node_id = attrs.get("id")
        file_name = attrs.get("fileName")

        if node_id and file_name:
            storage_nodes[node_id] = file_name

        if node.tag == "Volume":
            refs = parse_references(attrs.get("references", ""))
            volume_nodes[node_id] = {
                "name": attrs.get("name", ""),
                "storage_id": refs.get("storage"),
            }

    return volume_nodes, storage_nodes


def decode_slicer_name(name):
    name = Path(name).name
    name = urllib.parse.unquote(name)
    name = urllib.parse.unquote(name)
    return name


def find_local_file(patient_dir, slicer_file_name):
    decoded = decode_slicer_name(slicer_file_name).lower()

    candidates = list(patient_dir.glob("*.nrrd"))

    def clean(s):
        return (
            s.lower()
            .replace("%20", " ")
            .replace("%3a", ":")
            .replace("%253a", ":")
            .replace("_", " ")
            .replace("-", " ")
            .replace("(", "")
            .replace(")", "")
            .replace(".", " ")
        )

    decoded_clean = clean(decoded)

    # exact-ish match
    for c in candidates:
        if clean(c.name) == decoded_clean:
            return c

    # contains match
    for c in candidates:
        cname = clean(c.name)
        if decoded_clean in cname or cname in decoded_clean:
            return c

    # keyword fallback
    keywords = decoded_clean.split()

    best_file = None
    best_score = 0

    for c in candidates:
        cname = clean(c.name)
        score = sum(1 for k in keywords if k in cname)

        if score > best_score:
            best_score = score
            best_file = c

    if best_file is not None and best_score >= 2:
        return best_file

    print(f"\nCould not match: {slicer_file_name}")
    print("Available NRRD files:")
    for c in candidates:
        print(f"  {c.name}")

    raise FileNotFoundError(f"Could not match file: {slicer_file_name}")


def index_inside(idx_zyx, shape):
    z, y, x = idx_zyx
    return 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]


def modality_from_volume_name(volume_name):
    name = volume_name.lower()

    if "lateral" in name or "xray" in name:
        return "xray"

    if "pd_tse" in name or "mri" in name:
        return "mri"

    return "unknown"


def process_patient(patient_dir):
    patient_id = patient_dir.name

    volume_nodes, storage_nodes = parse_mrml(patient_dir)

    mrk_files = sorted(patient_dir.glob("*.mrk.json"))
    rows = []

    for mrk_file in mrk_files:
        points = load_points(mrk_file)

        if not points:
            continue

        node_id = points[0]["associatedNodeID"]
        volume_info = volume_nodes.get(node_id)

        if volume_info is None:
            print(f"Warning {patient_id}: no volume node for {node_id}")
            continue

        storage_file = storage_nodes.get(volume_info["storage_id"])

        if storage_file is None:
            print(f"Warning {patient_id}: no storage file for {node_id}")
            continue

        image_path = find_local_file(patient_dir, storage_file)
        image = sitk.ReadImage(str(image_path))
        volume = sitk.GetArrayFromImage(image)

        modality = modality_from_volume_name(volume_info["name"])

        for p in points:
            physical = [float(v) for v in p["position"]]

            try:
                idx_xyz = image.TransformPhysicalPointToIndex(physical)
                idx_zyx = np.array([idx_xyz[2], idx_xyz[1], idx_xyz[0]], dtype=int)
                inside = index_inside(idx_zyx, volume.shape)

                rows.append({
                    "patient_id": patient_id,
                    "modality": modality,
                    "volume_name": volume_info["name"],
                    "volume_file": image_path.name,
                    "mrk_file": mrk_file.name,
                    "point_index": p["point_index"],
                    "label": p["label"],

                    "physical_x": physical[0],
                    "physical_y": physical[1],
                    "physical_z": physical[2],

                    "index_x": int(idx_xyz[0]),
                    "index_y": int(idx_xyz[1]),
                    "index_z": int(idx_xyz[2]),

                    "shape_x": int(volume.shape[2]),
                    "shape_y": int(volume.shape[1]),
                    "shape_z": int(volume.shape[0]),

                    "inside_image": inside,
                })

            except Exception as e:
                rows.append({
                    "patient_id": patient_id,
                    "modality": modality,
                    "volume_name": volume_info["name"],
                    "volume_file": image_path.name,
                    "mrk_file": mrk_file.name,
                    "point_index": p["point_index"],
                    "label": p["label"],
                    "physical_x": physical[0],
                    "physical_y": physical[1],
                    "physical_z": physical[2],
                    "index_x": None,
                    "index_y": None,
                    "index_z": None,
                    "shape_x": int(volume.shape[2]),
                    "shape_y": int(volume.shape[1]),
                    "shape_z": int(volume.shape[0]),
                    "inside_image": False,
                    "error": str(e),
                })

    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted([p for p in EXTRACTED_DIR.iterdir() if p.is_dir()])
    all_rows = []

    print(f"Found {len(patient_dirs)} patients.")

    for patient_dir in patient_dirs:
        try:
            rows = process_patient(patient_dir)
            all_rows.extend(rows)
        except Exception as e:
            print(f"Error patient {patient_dir.name}: {e}")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n=============================")
    print("LANDMARK INDEX DATASET")
    print("=============================")
    print(f"Rows: {len(df)}")
    print(f"Patients: {df['patient_id'].nunique()}")
    print(f"Saved to: {OUTPUT_CSV}")

    print("\nModality counts:")
    print(df["modality"].value_counts())

    print("\nInside image counts:")
    print(df["inside_image"].value_counts())

    print("\nPoints per patient/modality:")
    print(
        df.groupby(["patient_id", "modality"])
        .size()
        .reset_index(name="num_points")
        .head(20)
    )


if __name__ == "__main__":
    main()