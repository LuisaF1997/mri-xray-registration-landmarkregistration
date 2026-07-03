from pathlib import Path
import xml.etree.ElementTree as ET


PATIENT_ID = "16"
PATIENT_DIR = Path("data/extracted") / PATIENT_ID

SEARCH_IDS = [
    "vtkMRMLScalarVolumeNode131",
    "vtkMRMLScalarVolumeNode132",
]


def main():
    mrml_file = next(PATIENT_DIR.glob("*.mrml"))
    tree = ET.parse(mrml_file)
    root = tree.getroot()

    print(f"Reading: {mrml_file}")

    for search_id in SEARCH_IDS:
        print("\n=============================")
        print(f"Searching for: {search_id}")
        print("=============================")

        found = False

        for node in root.iter():
            attrs = node.attrib

            if search_id in str(attrs):
                found = True
                print(f"\nTAG: {node.tag}")
                for key, value in attrs.items():
                    print(f"{key}: {value}")

        if not found:
            print("Not found.")


if __name__ == "__main__":
    main()