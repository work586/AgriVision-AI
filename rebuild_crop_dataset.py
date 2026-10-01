from pathlib import Path
import shutil
import random

ROOT = Path(__file__).resolve().parent

PLANTVILLAGE = ROOT / "datasets" / "PlantVillage"
DISEASE = ROOT / "datasets" / "processed" / "disease"
OUTPUT = ROOT / "datasets" / "processed" / "crop"

SEED = 42
random.seed(SEED)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff"
}

NK_MAPPING = {
    "black_gram": "Black_Gram",
    "chilli": "Chilli",
    "coconut": "Coconut",
    "cotton": "Cotton",
    "groundnut": "Groundnut",
    "guava": "Guava",
    "lemon": "Lemon",
    "mango": "Mango",
    "rice": "Rice",
    "sugarcane": "Sugarcane",
}

PV_MAPPING = {
    "Apple": "Apple",
    "Blueberry": "Blueberry",
    "Cherry_(including_sour)": "Cherry",
    "Corn_(maize)": "Corn",
    "Grape": "Grape",
    "Orange": "Orange",
    "Peach": "Peach",
    "Pepper,_bell": "Pepper_Bell",
    "Potato": "Potato",
    "Raspberry": "Raspberry",
    "Soybean": "Soybean",
    "Squash": "Squash",
    "Strawberry": "Strawberry",
    "Tomato": "Tomato",
}


def is_image(path):
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_images(folder):
    if not folder.exists():
        return []

    return [
        p for p in folder.rglob("*")
        if is_image(p)
    ]


def copy_image(src, output_dir, filename):
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, output_dir / filename)


def clear_output():
    if OUTPUT.exists():
        print("Removing existing processed crop dataset...")
        shutil.rmtree(OUTPUT)

    OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# PLANTVILLAGE
# ============================================================

def process_plantvillage():

    print("\n========================================")
    print("PROCESSING PLANTVILLAGE")
    print("========================================")

    train_root = PLANTVILLAGE / "train"
    val_root = PLANTVILLAGE / "val"

    total = 0

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_class_dirs = [
        p for p in train_root.iterdir()
        if p.is_dir()
    ]

    # Group disease folders by crop
    crop_images = {}

    for class_dir in train_class_dirs:

        class_name = class_dir.name

        if "___" not in class_name:
            print(f"Skipping unexpected folder: {class_name}")
            continue

        original_crop = class_name.split("___", 1)[0]

        if original_crop not in PV_MAPPING:
            continue

        final_crop = PV_MAPPING[original_crop]

        crop_images.setdefault(final_crop, [])

        crop_images[final_crop].extend(
            collect_images(class_dir)
        )

    # --------------------------------------------------------
    # Split PlantVillage TRAIN into 85/15
    # --------------------------------------------------------

    for final_crop, images in sorted(crop_images.items()):

        random.shuffle(images)

        split_index = int(len(images) * 0.85)

        train_images = images[:split_index]
        val_images = images[split_index:]

        train_out = OUTPUT / "train" / final_crop
        val_out = OUTPUT / "val" / final_crop

        for i, src in enumerate(train_images):

            filename = (
                f"plantvillage_{final_crop}_"
                f"train_{i:06d}"
                f"{src.suffix.lower()}"
            )

            copy_image(src, train_out, filename)

        for i, src in enumerate(val_images):

            filename = (
                f"plantvillage_{final_crop}_"
                f"val_{i:06d}"
                f"{src.suffix.lower()}"
            )

            copy_image(src, val_out, filename)

        # ----------------------------------------------------
        # PlantVillage ORIGINAL VALIDATION → TEST
        # ----------------------------------------------------

        test_images = []

        for class_dir in val_root.iterdir():

            if not class_dir.is_dir():
                continue

            class_name = class_dir.name

            if "___" not in class_name:
                continue

            original_crop = class_name.split("___", 1)[0]

            if original_crop not in PV_MAPPING:
                continue

            if PV_MAPPING[original_crop] != final_crop:
                continue

            test_images.extend(
                collect_images(class_dir)
            )

        test_out = OUTPUT / "test" / final_crop

        for i, src in enumerate(test_images):

            filename = (
                f"plantvillage_{final_crop}_"
                f"test_{i:06d}"
                f"{src.suffix.lower()}"
            )

            copy_image(src, test_out, filename)

        crop_total = (
            len(train_images)
            + len(val_images)
            + len(test_images)
        )

        total += crop_total

        print(
            f"{final_crop:<20}"
            f"train={len(train_images):>5} "
            f"val={len(val_images):>5} "
            f"test={len(test_images):>5} "
            f"total={crop_total:>5}"
        )

    print(f"\nPlantVillage processed: {total}")


# ============================================================
# NORTH KARNATAKA
# ============================================================

def process_north_karnataka():

    print("\n========================================")
    print("PROCESSING NORTH KARNATAKA")
    print("========================================")

    total = 0

    for source_crop, final_crop in NK_MAPPING.items():

        print(f"\nProcessing {final_crop}")

        for split in ["train", "val", "test"]:

            source_dir = DISEASE / source_crop / split

            if not source_dir.exists():
                print(
                    f"WARNING: missing {source_dir}"
                )
                continue

            images = collect_images(source_dir)

            output_dir = (
                OUTPUT / split / final_crop
            )

            for i, src in enumerate(images):

                filename = (
                    f"north_karnataka_"
                    f"{final_crop}_"
                    f"{split}_"
                    f"{i:06d}"
                    f"{src.suffix.lower()}"
                )

                copy_image(
                    src,
                    output_dir,
                    filename
                )

            print(
                f"{final_crop:<20}"
                f"{split:<6}"
                f"{len(images):>6}"
            )

            total += len(images)

    print(
        f"\nNorth Karnataka processed: {total}"
    )


# ============================================================
# AUDIT
# ============================================================

def count_images(folder):

    if not folder.exists():
        return 0

    return sum(
        1
        for p in folder.rglob("*")
        if is_image(p)
    )


def final_audit():

    print("\n========================================")
    print("FINAL DATASET AUDIT")
    print("========================================")

    grand_total = 0

    for split in ["train", "val", "test"]:

        count = count_images(
            OUTPUT / split
        )

        print(
            f"{split:<10}: {count}"
        )

        grand_total += count

    print("----------------------------------------")
    print(
        f"TOTAL     : {grand_total}"
    )
    print("========================================")

    if grand_total == 83615:

        print(
            "SUCCESS: Dataset contains exactly "
            "83,615 images."
        )

    else:

        print(
            f"WARNING: Expected 83,615 images "
            f"but found {grand_total}."
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "Crop dataset rebuild started."
    )

    print(
        f"Project root: {ROOT}"
    )

    if not PLANTVILLAGE.exists():
        raise FileNotFoundError(
            f"PlantVillage not found: "
            f"{PLANTVILLAGE}"
        )

    if not DISEASE.exists():
        raise FileNotFoundError(
            f"Disease dataset not found: "
            f"{DISEASE}"
        )

    print("\nSource datasets:")
    print(
        f"PlantVillage : {PLANTVILLAGE}"
    )
    print(
        f"Disease      : {DISEASE}"
    )
    print(
        f"Output       : {OUTPUT}"
    )

    clear_output()

    process_plantvillage()

    process_north_karnataka()

    final_audit()

    print("\nDONE.")