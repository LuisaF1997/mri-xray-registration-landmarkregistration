from pathlib import Path
import json

PATIENT_ID = "16"
PATIENT_DIR = Path("data/extracted") / PATIENT_ID


def inspect_mrk(file):
    print("\n=============================")
    print(file.name)
    print("=============================")

    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for markup in data.get("markups", []):
        print("coordinateSystem:", markup.get("coordinateSystem"))
        print("locked:", markup.get("locked"))
        print("labelFormat:", markup.get("labelFormat"))

        for cp in markup.get("controlPoints", []):
            print(
                cp.get("label"),
                "position:",
                cp.get("position"),
                "orientation:",
                cp.get("orientation"),
                "associatedNodeID:",
                cp.get("associatedNodeID"),
            )


def main():
    print(f"Patient folder: {PATIENT_DIR}")

    print("\nMRK files:")
    for f in PATIENT_DIR.glob("*.mrk.json"):
        inspect_mrk(f)

    print("\nNRRD files:")
    for f in PATIENT_DIR.glob("*.nrrd"):
        print(f.name)

    print("\nTransform files:")
    for f in PATIENT_DIR.glob("*.h5"):
        print(f.name)

    print("\nOther files:")
    for f in PATIENT_DIR.iterdir():
        if f.is_file() and not f.name.endswith((".mrk.json", ".nrrd", ".h5")):
            print(f.name)

    print("\nMRML files:")
    for f in PATIENT_DIR.glob("*.mrml"):
        print(f.name)


if __name__ == "__main__":
    main()