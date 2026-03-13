import argparse
import os
import random
from datetime import datetime

import numpy as np
from PIL import Image

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
LABEL_SUFFIXES = ("labels", "_labels", "-labels", "label", "_label", "-label")


def _list_image_files(folder_path):
    if not os.path.isdir(folder_path):
        return []

    return sorted(
        [
            name
            for name in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, name))
            and name.lower().endswith(IMAGE_EXTS)
        ]
    )


def _is_probable_label(stem_name):
    stem_lower = stem_name.lower()
    return any(stem_lower.endswith(suffix) for suffix in LABEL_SUFFIXES)


def _find_label_path(image_stem, stem_to_path):
    candidates = [
        f"{image_stem}labels",
        f"{image_stem}_labels",
        f"{image_stem}-labels",
        f"{image_stem}label",
        f"{image_stem}_label",
        f"{image_stem}-label",
    ]

    for candidate in candidates:
        matched = stem_to_path.get(candidate.lower())
        if matched:
            return matched

    return None


def find_image_label_pairs(orthoimage_dir):
    files = _list_image_files(orthoimage_dir)
    stem_to_path = {
        os.path.splitext(file_name)[0].lower(): os.path.join(orthoimage_dir, file_name)
        for file_name in files
    }

    pairs = []
    unmatched_images = []

    for file_name in files:
        image_stem = os.path.splitext(file_name)[0]
        image_path = os.path.join(orthoimage_dir, file_name)

        if _is_probable_label(image_stem):
            continue

        label_path = _find_label_path(image_stem, stem_to_path)
        if label_path and os.path.normcase(label_path) != os.path.normcase(image_path):
            pairs.append((image_path, label_path))
        else:
            unmatched_images.append(image_path)

    return pairs, unmatched_images


def _sliding_positions(length, patch_size, stride):
    if length <= patch_size:
        return [0]

    positions = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def process_pair(image_path, label_path, target_height, patch_size, stride, white_threshold):
    """Return a list of (image_patch, label_patch) pairs for one image/label pair."""
    with Image.open(image_path) as image_raw, Image.open(label_path) as label_raw:
        image = image_raw.convert("L")
        label = label_raw.convert("L")

        scale = target_height / max(1, image.height)
        new_width = max(patch_size, int(round(image.width * scale)))
        image = image.resize((new_width, target_height), Image.BILINEAR)
        label = label.resize((new_width, target_height), Image.NEAREST)

        patches = []
        x_positions = _sliding_positions(new_width, patch_size, stride)
        y_positions = _sliding_positions(target_height, patch_size, stride)

        for x in x_positions:
            for y in y_positions:
                img_patch = image.crop((x, y, x + patch_size, y + patch_size))
                lbl_patch = label.crop((x, y, x + patch_size, y + patch_size))

                if white_threshold is not None:
                    white_ratio = float(np.mean(np.array(lbl_patch) > 250))
                    if white_ratio > white_threshold:
                        continue

                patches.append((img_patch, lbl_patch))

        return patches


def _build_dataset_paths(dataset_output_root):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_root = os.path.join(dataset_output_root, timestamp, "membrane")

    paths = {
        "dataset_root": dataset_root,
        "test_dir": os.path.join(dataset_root, "test"),
        "train_img_dir": os.path.join(dataset_root, "train", "image"),
        "train_lbl_dir": os.path.join(dataset_root, "train", "label"),
        "val_img_dir": os.path.join(dataset_root, "validation", "image"),
        "val_lbl_dir": os.path.join(dataset_root, "validation", "label"),
    }

    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    return paths


def _save_patch_pairs(patch_pairs, image_dir, label_dir):
    for index, (img_patch, lbl_patch) in enumerate(patch_pairs):
        file_name = f"{index:06d}.png"
        img_patch.save(os.path.join(image_dir, file_name))
        lbl_patch.save(os.path.join(label_dir, file_name))


def _copy_unmatched_to_test(unmatched_images, test_dir):
    for image_path in unmatched_images:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(test_dir, f"{stem}.png")

        suffix_index = 1
        while os.path.exists(out_path):
            out_path = os.path.join(test_dir, f"{stem}_{suffix_index}.png")
            suffix_index += 1

        with Image.open(image_path) as raw_image:
            raw_image.convert("L").save(out_path)


def create_dataset(
    orthoimage_dir,
    dataset_output_root,
    validation_split=0.2,
    target_height=256,
    patch_size=256,
    stride=128,
    white_threshold=0.98,
    seed=42,
    copy_unmatched_to_test=True,
):
    pairs, unmatched_images = find_image_label_pairs(orthoimage_dir)
    if not pairs:
        raise RuntimeError(
            "No matching image/label pairs found in orthoimages. "
            "Expected labels named like <orthoimage>labels.* or <orthoimage>_labels.*"
        )

    print(f"Found {len(pairs)} image/label pairs. Generating patches...")

    all_patches = []
    for image_path, label_path in pairs:
        all_patches.extend(
            process_pair(
                image_path=image_path,
                label_path=label_path,
                target_height=target_height,
                patch_size=patch_size,
                stride=stride,
                white_threshold=white_threshold,
            )
        )

    if not all_patches:
        raise RuntimeError("No patches were generated. Check image sizes or white-threshold filtering.")

    rng = random.Random(seed)
    rng.shuffle(all_patches)

    total_patches = len(all_patches)
    val_count = int(total_patches * validation_split)
    if total_patches > 1 and validation_split > 0 and val_count == 0:
        val_count = 1
    if val_count >= total_patches:
        val_count = total_patches - 1

    validation_patches = all_patches[:val_count]
    training_patches = all_patches[val_count:]

    paths = _build_dataset_paths(dataset_output_root)
    _save_patch_pairs(training_patches, paths["train_img_dir"], paths["train_lbl_dir"])
    _save_patch_pairs(validation_patches, paths["val_img_dir"], paths["val_lbl_dir"])

    if copy_unmatched_to_test and unmatched_images:
        _copy_unmatched_to_test(unmatched_images, paths["test_dir"])

    summary = {
        "dataset_root": paths["dataset_root"],
        "pairs_found": len(pairs),
        "unmatched_images": len(unmatched_images),
        "total_patches": total_patches,
        "training_patches": len(training_patches),
        "validation_patches": len(validation_patches),
    }

    print("\nDataset creation complete.")
    print(f"Output: {summary['dataset_root']}")
    print(f"Total patches: {summary['total_patches']}")
    print(f"Training patches: {summary['training_patches']}")
    print(f"Validation patches: {summary['validation_patches']}")
    print(f"Unmatched orthoimages copied to test/: {summary['unmatched_images']}")

    return summary


def createDataSet(*args, **kwargs):
    """Backward compatible alias used by older scripts."""
    return create_dataset(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Create U-Net dataset from orthoimages and labels")
    parser.add_argument("orthoimage_dir", help="Folder containing orthoimages and labels")
    parser.add_argument("dataset_output_root", help="Root folder where timestamped dataset folder is created")
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--target-height", type=int, default=512)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--white-threshold", type=float, default=0.98)
    args = parser.parse_args()

    create_dataset(
        orthoimage_dir=args.orthoimage_dir,
        dataset_output_root=args.dataset_output_root,
        validation_split=args.validation_split,
        target_height=args.target_height,
        patch_size=args.patch_size,
        stride=args.stride,
        white_threshold=args.white_threshold,
    )


if __name__ == "__main__":
    main()