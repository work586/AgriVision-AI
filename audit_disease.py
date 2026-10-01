from pathlib import Path

ROOT = Path("datasets/processed/disease")

print("=" * 70)
print("DISEASE DATASET AUDIT")
print("=" * 70)

grand_total = 0

for crop_dir in sorted(ROOT.iterdir()):

    if not crop_dir.is_dir():
        continue

    print()
    print(crop_dir.name.upper())
    print("-" * 50)

    crop_total = 0

    for split in ["train", "val", "test"]:

        split_dir = crop_dir / split

        if not split_dir.exists():
            print(f"{split}: MISSING")
            continue

        split_total = 0

        for disease_dir in sorted(split_dir.iterdir()):

            if not disease_dir.is_dir():
                continue

            count = sum(
                1
                for f in disease_dir.iterdir()
                if f.is_file()
            )

            split_total += count

            print(
                f"  {split:5s} | "
                f"{disease_dir.name:30s} | "
                f"{count:5d}"
            )

        print(f"  {split:5s} TOTAL: {split_total}")

        crop_total += split_total

    print(f"  CROP TOTAL: {crop_total}")

    grand_total += crop_total


print()
print("=" * 70)
print(f"GRAND TOTAL: {grand_total}")
print("=" * 70)