from pathlib import Path
import shutil
import random
import re

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "datasets" / "PlantVillage"
OUTPUT = ROOT / "datasets" / "processed" / "disease" / "plantvillage"

TRAIN_SOURCE = SOURCE / "train"
VAL_SOURCE = SOURCE / "val"

TEST_RATIO = 0.15
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ============================================================
# CLEAN CLASS NAME
# ============================================================

def clean_class_name(name):
    """
    Convert PlantVillage class names into clean folder names.
    """

    name = name.strip()

    # Remove trailing underscores/spaces
    name = re.sub(r"[_\s]+$", "", name)

    # Replace problematic characters
    name = name.replace(",", "_")
    name = name.replace("(", "")
    name = name.replace(")", "")
    name = name.replace("/", "_")
    name = name.replace("\\", "_")
    name = name.replace(" ", "_")

    # Remove repeated underscores
    name = re.sub(r"_+", "_", name)

    # Remove trailing underscore again
    name = name.strip("_")

    return name


# ============================================================
# GET CROP NAME
# ============================================================

def get_crop_name(class_name):
    """
    PlantVillage class format:

        Apple___Apple_scab
        Apple___healthy
        Tomato___Early_blight

    Crop name is everything before ___.
    """

    if "___" not in class_name:
        return None

    crop, _ = class_name.split("___", 1)

    crop = clean_class_name(crop)

    # Normalize PlantVillage crop names
    crop_mapping = {
        "Apple": "apple",
        "Blueberry": "blueberry",
        "Cherry_including_sour": "cherry_(including_sour)",
        "Corn_maize": "corn",
        "Grape": "grape",
        "Orange": "orange",
        "Peach": "peach",
        "Pepper_bell": "pepper_bell",
        "Potato": "potato",
        "Raspberry": "raspberry",
        "Soybean": "soybean",
        "Squash": "squash",
        "Strawberry": "strawberry",
        "Tomato": "tomato",
    }

    return crop_mapping.get(crop, crop.lower())


# ============================================================
# GET DISEASE NAME
# ============================================================

def get_disease_name(class_name):
    """
    Get everything after ___.
    """

    if "___" not in class_name:
        return None

    _, disease = class_name.split("___", 1)

    disease = clean_class_name(disease)

    # Normalize healthy labels
    if disease.lower() == "healthy":
        return "healthy"

    if disease.lower() == "soybean_healthy":
        return "healthy"

    return disease


# ============================================================
# COPY IMAGES
# ============================================================

def copy_images(images, destination):
    destination.mkdir(parents=True, exist_ok=True)

    for image in images:
        target = destination / image.name

        # Avoid filename collision
        counter = 1

        while target.exists():
            target = destination / f"{image.stem}_{counter}{image.suffix}"
            counter += 1

        shutil.copy2(image, target)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PLANTVILLAGE DISEASE DATASET PREPARATION")
    print("=" * 70)

    print(f"Source : {SOURCE}")
    print(f"Output : {OUTPUT}")
    print()

    if not TRAIN_SOURCE.exists():
        raise FileNotFoundError(
            f"PlantVillage train folder not found:\n{TRAIN_SOURCE}"
        )

    if not VAL_SOURCE.exists():
        raise FileNotFoundError(
            f"PlantVillage val folder not found:\n{VAL_SOURCE}"
        )

    # --------------------------------------------------------
    # DELETE OLD GENERATED DATA
    # --------------------------------------------------------

    if OUTPUT.exists():
        print("Removing old processed PlantVillage dataset...")
        shutil.rmtree(OUTPUT)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # FIND CLASSES
    # --------------------------------------------------------

    train_classes = {
        d.name: d
        for d in TRAIN_SOURCE.iterdir()
        if d.is_dir() and "___" in d.name
    }

    val_classes = {
        d.name: d
        for d in VAL_SOURCE.iterdir()
        if d.is_dir() and "___" in d.name
    }

    all_classes = sorted(set(train_classes) | set(val_classes))

    print(f"Detected classes: {len(all_classes)}")
    print()

    # --------------------------------------------------------
    # GROUP CLASSES BY CROP
    # --------------------------------------------------------

    crop_classes = {}

    for class_name in all_classes:

        crop = get_crop_name(class_name)
        disease = get_disease_name(class_name)

        if crop is None or disease is None:
            continue

        crop_classes.setdefault(crop, []).append(
            (class_name, disease)
        )

    print(f"Detected crops: {len(crop_classes)}")
    print()

    # --------------------------------------------------------
    # PROCESS EACH CROP
    # --------------------------------------------------------

    total_train = 0
    total_val = 0
    total_test = 0

    for crop in sorted(crop_classes):

        print("-" * 70)
        print(f"CROP: {crop}")

        crop_train_count = 0
        crop_val_count = 0
        crop_test_count = 0

        for class_name, disease in sorted(crop_classes[crop]):

            train_dir = train_classes.get(class_name)
            val_dir = val_classes.get(class_name)

            # ------------------------------------------------
            # SOURCE TRAIN IMAGES
            # ------------------------------------------------

            train_images = []

            if train_dir and train_dir.exists():

                train_images = [
                    p for p in train_dir.iterdir()
                    if p.is_file()
                    and p.suffix.lower() in {
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".bmp",
                        ".webp"
                    }
                ]

            # ------------------------------------------------
            # SHUFFLE
            # ------------------------------------------------

            random.shuffle(train_images)

            # ------------------------------------------------
            # HOLD OUT TEST SET
            #
            # IMPORTANT:
            # Test images are REMOVED from train.
            # This prevents data leakage.
            # ------------------------------------------------

            if len(train_images) >= 2:

                test_count = max(
                    1,
                    int(len(train_images) * TEST_RATIO)
                )

                test_images = train_images[:test_count]
                remaining_train = train_images[test_count:]

            else:

                test_images = []
                remaining_train = train_images

            # ------------------------------------------------
            # DESTINATION PATHS
            # ------------------------------------------------

            train_dest = (
                OUTPUT
                / crop
                / "train"
                / disease
            )

            val_dest = (
                OUTPUT
                / crop
                / "val"
                / disease
            )

            test_dest = (
                OUTPUT
                / crop
                / "test"
                / disease
            )

            # ------------------------------------------------
            # COPY TRAIN
            # ------------------------------------------------

            copy_images(
                remaining_train,
                train_dest
            )

            # ------------------------------------------------
            # COPY TEST
            # ------------------------------------------------

            copy_images(
                test_images,
                test_dest
            )

            # ------------------------------------------------
            # COPY ORIGINAL VALIDATION
            # ------------------------------------------------

            val_images = []

            if val_dir and val_dir.exists():

                val_images = [
                    p for p in val_dir.iterdir()
                    if p.is_file()
                    and p.suffix.lower() in {
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".bmp",
                        ".webp"
                    }
                ]

            copy_images(
                val_images,
                val_dest
            )

            # ------------------------------------------------
            # COUNTS
            # ------------------------------------------------

            train_count = len(remaining_train)
            test_count_actual = len(test_images)
            val_count = len(val_images)

            crop_train_count += train_count
            crop_val_count += val_count
            crop_test_count += test_count_actual

            total_train += train_count
            total_val += val_count
            total_test += test_count_actual

            print(
                f"  {disease:<45} "
                f"train={train_count:<5} "
                f"val={val_count:<5} "
                f"test={test_count_actual:<5}"
            )

        print(
            f"  TOTAL {crop}: "
            f"train={crop_train_count}, "
            f"val={crop_val_count}, "
            f"test={crop_test_count}"
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PREPARATION COMPLETE")
    print("=" * 70)

    print(f"Total train images : {total_train}")
    print(f"Total validation   : {total_val}")
    print(f"Total test images  : {total_test}")

    print()
    print("IMPORTANT:")
    print("Test images were held out from the original training set.")
    print("Therefore train/test leakage has been removed.")

    print()
    print(f"Output folder:")
    print(OUTPUT)


if __name__ == "__main__":
    main()