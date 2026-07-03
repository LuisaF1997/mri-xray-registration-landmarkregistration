from pathlib import Path
import zipfile
import json
import shutil


RAW_DIR = Path("data/raw")
EXTRACTED_DIR = Path("data/extracted")


def extract_mrb(mrb_path: Path, patient_id: str):
    patient_out = EXTRACTED_DIR / patient_id
    patient_out.mkdir(parents=True, exist_ok=True)

    temp_dir = patient_out / "_unzipped"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    temp_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(mrb_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    return temp_dir, patient_out


def find_files(temp_dir: Path):
    mrk_files = list(temp_dir.rglob("*.mrk.json"))
    nrrd_files = list(temp_dir.rglob("*.nrrd"))
    h5_files = list(temp_dir.rglob("*.h5"))
    mrml_files = list(temp_dir.rglob("*.mrml"))

    return mrk_files, nrrd_files, h5_files, mrml_files


def load_landmarks(mrk_path: Path):
    with open(mrk_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    points = []

    markups = data.get("markups", [])

    for markup in markups:
        for cp in markup.get("controlPoints", []):
            points.append({
                "label": cp.get("label", ""),
                "position": cp.get("position", []),
                "associatedNodeID": cp.get("associatedNodeID", ""),
            })

    return points


def process_patient(patient_folder: Path):
    patient_id = patient_folder.name
    mrb_files = list(patient_folder.glob("*.mrb"))

    if len(mrb_files) == 0:
        print(f"Skipping {patient_id}: no MRB file found")
        return

    mrb_path = mrb_files[0]

    print("\n-----------------------------")
    print(f"Patient: {patient_id}")
    print(f"MRB: {mrb_path}")

    temp_dir, patient_out = extract_mrb(mrb_path, patient_id)
    mrk_files, nrrd_files, h5_files, mrml_files = find_files(temp_dir)

    print(f"Found landmark files: {len(mrk_files)}")
    print(f"Found NRRD files: {len(nrrd_files)}")
    print(f"Found transform files: {len(h5_files)}")
    print(f"Found MRML files: {len(mrml_files)}")

    landmark_summary = {}

    # Copy and summarize landmark files
    for mrk_file in mrk_files:
        points = load_landmarks(mrk_file)
        landmark_summary[mrk_file.name] = points

        out_file = patient_out / mrk_file.name
        shutil.copy(mrk_file, out_file)

        print(f"  {mrk_file.name}: {len(points)} points")

    # Copy NRRD volume files
    for nrrd_file in nrrd_files:
        shutil.copy(nrrd_file, patient_out / nrrd_file.name)

    # Copy transform files
    for h5_file in h5_files:
        shutil.copy(h5_file, patient_out / h5_file.name)

    # Copy MRML scene files
    for mrml_file in mrml_files:
        shutil.copy(mrml_file, patient_out / mrml_file.name)

    # Save landmark summary
    summary_path = patient_out / "landmark_summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(landmark_summary, f, indent=4)

    print(f"Saved extracted files to: {patient_out}")


def main():
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    patient_folders = sorted([p for p in RAW_DIR.iterdir() if p.is_dir()])

    print(f"Found {len(patient_folders)} patient folders.")

    for patient_folder in patient_folders:
        process_patient(patient_folder)


if __name__ == "__main__":
    main()