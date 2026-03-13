# U-Net Pointcloud Tools

Utilities for processing point clouds and preparing U-Net training data.

This repository supports:
- Converting `.ply` point clouds to `.las`
- Removing isolated points from LAS files
- Creating orthoimages from denoised LAS files
- Building a timestamped U-Net dataset from orthoimages and labels

## Project Layout

The unified command flow expects this workspace structure:

```text
pointcloud/
  ply/
  las/
    denoised/
images/
  orthoimages/
  unet_training_data/
src/
  main.py
  convert_to_las.py
  remove_isolated_points.py
  pointcloud2orthoimage.py
  create_dataset.py
```

## Setup

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Unified CLI (Recommended)

Run all pipeline stages through:

```powershell
python src/main.py <command> [options]
```

Use help:

```powershell
python src/main.py -h
python src/main.py <command> -h
```

### 1. `convert_to_las`

Converts all `.ply` files in `pointcloud/ply/` into `.las` files in `pointcloud/las/`.
Existing LAS files are skipped.

Options:
- `--workspace-root` (default: project root)
- `--chunk-size` (default: `1000000`)

Example:

```powershell
python src/main.py convert_to_las --chunk-size 1000000
```

### 2. `remove_isolated_points`

Denoises each LAS file in `pointcloud/las/` and writes results to
`pointcloud/las/denoised/`. Existing denoised outputs are skipped.

Options:
- `--workspace-root` (default: project root)
- `--voxel-size` (default: `0.05`)
- `--radius` (default: `0.15`)
- `--min-neighbors` (default: `8`)
- `--chunk-size` (default: `1000000`)

Example:

```powershell
python src/main.py remove_isolated_points --voxel-size 0.05 --radius 0.15 --min-neighbors 8
```

### 3. `create_orthoimages`

Reads denoised LAS files from `pointcloud/las/denoised/` and writes orthoimages to
`images/orthoimages/` as `<stem>RGB.jpg`.

Options:
- `--workspace-root` (default: project root)
- `--downsample` (default: `10`)
- `--gsd-mm-per-px` (default: `5.0`)
- `--overwrite` (default: off)

Example:

```powershell
python src/main.py create_orthoimages --downsample 10 --gsd-mm-per-px 5.0
```

### 4. `create_dataset`

Builds a timestamped dataset under:
`images/unet_training_data/<timestamp>/membrane/`

Expected folder outputs:
- `train/image`
- `train/label`
- `validation/image`
- `validation/label`
- `test`

Labels are matched to images using these suffix patterns:
- `labels`, `_labels`, `-labels`, `label`, `_label`, `-label`

Options:
- `--workspace-root` (default: project root)
- `--validation-split` (default: `0.2`)
- `--target-height` (default: `256`)
- `--patch-size` (default: `256`)
- `--stride` (default: `128`)
- `--white-threshold` (default: `0.98`)
- `--seed` (default: `42`)
- `--no-copy-unmatched-to-test` (default: off)

Example:

```powershell
python src/main.py create_dataset --validation-split 0.2 --patch-size 256 --stride 128
```

## Standalone Script Commands

These scripts also expose direct CLI entry points:

### `src/remove_isolated_points.py`

```powershell
python src/remove_isolated_points.py <input> [--output <path>] [--suffix _denoised] [--voxel-size 0.05] [--radius 0.15] [--min-neighbors 8] [--chunk-size 1000000]
```

Notes:
- `<input>` can be a single `.las` file or a directory of `.las` files.
- If `--output` is omitted, output is written next to each input as `*_denoised.las`.

### `src/create_dataset.py`

```powershell
python src/create_dataset.py <orthoimage_dir> <dataset_output_root> [--validation-split 0.2] [--target-height 512] [--patch-size 512] [--stride 256] [--white-threshold 0.98]
```

### Module-Only (No CLI Parser)

- `src/convert_to_las.py` provides function `convert_to_las(...)`.
- `src/pointcloud2orthoimage.py` provides functions like `main2(...)` and `p2o_main(...)`.

These are used by `src/main.py` and can also be imported from Python code.

## Typical End-to-End Run

From project root:

```powershell
python src/main.py convert_to_las
python src/main.py remove_isolated_points
python src/main.py create_orthoimages
python src/main.py create_dataset
```
