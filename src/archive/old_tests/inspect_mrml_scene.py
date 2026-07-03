from pathlib import Path
import xml.etree.ElementTree as ET


PATIENT_ID = "16"
PATIENT_DIR = Path("data/extracted") / PATIENT_ID


def main():
    mrml_files = list(PATIENT_DIR.glob("*.mrml"))

    if not mrml_files:
        raise FileNotFoundError(f"No MRML file found in {PATIENT_DIR}")

    mrml_file = mrml_files[0]
    print(f"Reading MRML: {mrml_file}")

    tree = ET.parse(mrml_file)
    root = tree.getroot()

    print("\n=============================")
    print("Scalar Volume Nodes")
    print("=============================")

    for node in root.iter():
        tag = node.tag

        if "ScalarVolume" in tag or node.attrib.get("class") == "vtkMRMLScalarVolumeNode":
            print("\nNode:")
            for key, value in node.attrib.items():
                if (
                    "id" in key.lower()
                    or "name" in key.lower()
                    or "storage" in key.lower()
                    or "transform" in key.lower()
                    or "file" in key.lower()
                ):
                    print(f"  {key}: {value}")

    print("\n=============================")
    print("Transform Nodes")
    print("=============================")

    for node in root.iter():
        tag = node.tag

        if "Transform" in tag or node.attrib.get("class") == "vtkMRMLTransformNode":
            print("\nNode:")
            for key, value in node.attrib.items():
                print(f"  {key}: {value}")

    print("\n=============================")
    print("Storage Nodes")
    print("=============================")

    for node in root.iter():
        if "Storage" in node.tag:
            print("\nNode:")
            for key, value in node.attrib.items():
                if (
                    "id" in key.lower()
                    or "name" in key.lower()
                    or "file" in key.lower()
                    or "uri" in key.lower()
                ):
                    print(f"  {key}: {value}")


if __name__ == "__main__":
    main()