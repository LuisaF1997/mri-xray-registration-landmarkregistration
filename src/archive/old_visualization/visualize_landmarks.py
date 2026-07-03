from pathlib import Path
import json
import xml.etree.ElementTree as ET
import urllib.parse

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk


PATIENT_ID = "16"
EXTRACTED_DIR = Path("data/extracted")
OUTPUT_DIR = Path("outputs/landmark_visualization")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize(img):
    p1, p99 = np.percentile(img, [1, 99])
    img = np.clip(img, p1, p99)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def load_points(mrk_file):
    with open(mrk_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    points = []
    for markup in data.get("markups", []):
        for cp in markup.get("controlPoints", []):
            points.append({
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
    transform_nodes = {}

    for node in root.iter():
        attrs = node.attrib
        tag = node.tag

        node_id = attrs.get("id")
        file_name = attrs.get("fileName")

        # Store every node with a fileName
        if node_id and file_name:
            storage_nodes[node_id] = file_name

        # Volume nodes
        if tag == "Volume":
            refs = parse_references(attrs.get("references", ""))
            volume_nodes[node_id] = {
                "name": attrs.get("name", ""),
                "storage_id": refs.get("storage"),
                "transform_id": refs.get("transform"),
            }

        # Transform nodes
        if "TransformNode" in tag:
            refs = parse_references(attrs.get("references", ""))
            transform_nodes[node_id] = {
                "name": attrs.get("name", ""),
                "storage_id": refs.get("storage"),
            }

    print("\nStorage nodes found:")
    for k, v in storage_nodes.items():
        print(f"{k}: {v}")

    return volume_nodes, storage_nodes, transform_nodes


def decode_slicer_name(name):
    name = Path(name).name
    name = urllib.parse.unquote(name)
    name = urllib.parse.unquote(name)
    return name


def find_local_file(patient_dir, slicer_file_name):
    if slicer_file_name is None:
        raise ValueError("slicer_file_name is None")

    decoded = decode_slicer_name(slicer_file_name).lower()

    candidates = list(patient_dir.glob("*"))

    for c in candidates:
        if c.name.lower() == decoded:
            return c

    for c in candidates:
        cname = c.name.lower()

        if "lateral" in decoded and "lateral" in cname:
            return c

        if "pd_tse" in decoded and "pd_tse" in cname:
            return c

        if "reg_xray" in decoded and "reg_xray" in cname:
            return c

        if "xraymri_registration_transform" in decoded and "xraymri_registration_transform" in cname:
            return c

        if "transform_40" in decoded and "transform_40" in cname:
            return c

    raise FileNotFoundError(f"Could not match file: {slicer_file_name}")


def load_transform(patient_dir, transform_id, storage_nodes, transform_nodes):
    if transform_id is None:
        return None

    tnode = transform_nodes.get(transform_id)
    if tnode is None:
        print(f"Warning: transform node not found: {transform_id}")
        return None

    storage_id = tnode.get("storage_id")
    storage_file = storage_nodes.get(storage_id)

    if storage_file is None:
        print(f"Warning: transform storage missing for {transform_id}")
        return None

    transform_path = find_local_file(patient_dir, storage_file)
    print(f"Loading transform {transform_id}: {transform_path}")

    return sitk.ReadTransform(str(transform_path))


def world_to_local(point, transform):
    point = [float(v) for v in point]

    if transform is None:
        return point

    return transform.GetInverse().TransformPoint(point)


def physical_to_index(image, point):
    idx_xyz = image.TransformPhysicalPointToIndex([float(v) for v in point])
    return np.array([idx_xyz[2], idx_xyz[1], idx_xyz[0]], dtype=int)


def index_inside(idx, shape):
    z, y, x = idx
    return 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]


def plot_landmarks(image_path, points, transform, title, output_path):
    image = sitk.ReadImage(str(image_path))
    volume = sitk.GetArrayFromImage(image).astype(np.float32)

    converted = []

    for p in points:
        try:
            local = world_to_local(p["position"], transform)
            idx = physical_to_index(image, local)
            inside = index_inside(idx, volume.shape)

            converted.append({
                "label": p["label"],
                "world": p["position"],
                "local": local,
                "index": idx,
                "inside": inside,
            })

        except Exception as e:
            converted.append({
                "label": p["label"],
                "world": p["position"],
                "local": None,
                "index": None,
                "inside": False,
                "error": str(e),
            })

    print("\n-----------------------------")
    print(title)
    print(f"Image: {image_path}")
    print(f"Shape z,y,x: {volume.shape}")
    print(f"Spacing: {image.GetSpacing()}")
    print(f"Origin: {image.GetOrigin()}")

    for p in converted:
        print(
            f"{p['label']} | world={p['world']} | local={p['local']} | "
            f"index={p['index']} | inside={p['inside']}"
        )

    valid = [p["index"] for p in converted if p["inside"]]

    if valid:
        z_mid, y_mid, x_mid = np.median(np.array(valid), axis=0).astype(int)
    else:
        z_mid = volume.shape[0] // 2
        y_mid = volume.shape[1] // 2
        x_mid = volume.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    views = [
        ("Axial", volume[z_mid, :, :]),
        ("Coronal", volume[:, y_mid, :]),
        ("Sagittal", volume[:, :, x_mid]),
    ]

    for ax, (view_name, img) in zip(axes, views):
        ax.imshow(normalize(img), cmap="gray")
        ax.set_title(view_name)

        for i, p in enumerate(converted, start=1):
            if not p["inside"]:
                continue

            z, y, x = p["index"]

            if view_name == "Axial":
                px, py = x, y
            elif view_name == "Coronal":
                px, py = x, z
            else:
                px, py = y, z

            ax.scatter(px, py, s=80)
            ax.text(px + 2, py + 2, str(i), fontsize=12, color="yellow")

        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    patient_dir = EXTRACTED_DIR / PATIENT_ID

    volume_nodes, storage_nodes, transform_nodes = parse_mrml(patient_dir)

    print("\nVolume nodes:")
    for k, v in volume_nodes.items():
        print(k, v)

    from_file = next(patient_dir.glob("*From*.mrk.json"))
    to_file = next(patient_dir.glob("*To*.mrk.json"))

    for set_name, mrk_file in [("from", from_file), ("to", to_file)]:
        points = load_points(mrk_file)
        node_id = points[0]["associatedNodeID"]

        volume_info = volume_nodes[node_id]
        storage_file = storage_nodes[volume_info["storage_id"]]
        image_path = find_local_file(patient_dir, storage_file)

        transform = load_transform(
            patient_dir,
            volume_info["transform_id"],
            storage_nodes,
            transform_nodes,
        )

        title = f"Patient {PATIENT_ID} - {set_name} landmarks on {volume_info['name']}"
        output_path = OUTPUT_DIR / f"{PATIENT_ID}_{set_name}_landmarks_corrected.png"

        plot_landmarks(
            image_path=image_path,
            points=points,
            transform=transform,
            title=title,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()