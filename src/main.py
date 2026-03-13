import argparse
import os
import sys
from pathlib import Path


DEFAULT_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


def ensure_project_folders(workspace_root):
	root = Path(workspace_root).resolve()

	folders = {
		"root": root,
		"ply_dir": root / "pointcloud" / "ply",
		"las_dir": root / "pointcloud" / "las",
		"denoised_dir": root / "pointcloud" / "las" / "denoised",
		"ortho_dir": root / "images" / "orthoimages",
		"dataset_dir": root / "images" / "unet_training_data",
	}

	for path in folders.values():
		if path == root:
			continue
		path.mkdir(parents=True, exist_ok=True)

	return folders


def _iter_las_files(folder_path):
	if not folder_path.exists():
		return []

	return sorted(
		[
			file_path
			for file_path in folder_path.iterdir()
			if file_path.is_file() and file_path.suffix.lower() == ".las"
		],
		key=lambda value: value.name.lower(),
	)


def cmd_convert_to_las(args):
	folders = ensure_project_folders(args.workspace_root)
	from convert_to_las import convert_to_las

	created = convert_to_las(
		input_dir=str(folders["ply_dir"]),
		output_dir=str(folders["las_dir"]),
		chunk_size=args.chunk_size,
		skip_existing=True,
	)

	print(f"Finished convert_to_las. New files created: {len(created)}")


def cmd_remove_isolated_points(args):
	folders = ensure_project_folders(args.workspace_root)
	from remove_isolated_points import remove_isolated_points

	las_files = _iter_las_files(folders["las_dir"])
	if not las_files:
		print(f"No LAS files found in {folders['las_dir']}")
		return

	created_count = 0
	skipped_count = 0

	for source_las in las_files:
		output_las = folders["denoised_dir"] / source_las.name
		if output_las.exists():
			print(f"Skipping existing denoised LAS: {output_las.name}")
			skipped_count += 1
			continue

		remove_isolated_points(
			input_las_path=str(source_las),
			output_las_path=str(output_las),
			voxel_size=args.voxel_size,
			radius=args.radius,
			min_neighbors=args.min_neighbors,
			chunk_size=args.chunk_size,
		)
		created_count += 1

	print(
		"Finished remove_isolated_points. "
		f"New files created: {created_count}, skipped existing: {skipped_count}"
	)


def _platform_flag():
	if os.name == "nt":
		return "win"
	if sys.platform == "darwin":
		return "mac"
	return "server"


def cmd_create_orthoimages(args):
	folders = ensure_project_folders(args.workspace_root)

	# Import lazily because pointcloud2orthoimage has heavy optional dependencies.
	from pointcloud2orthoimage import main2

	denoised_files = _iter_las_files(folders["denoised_dir"])
	if not denoised_files:
		print(f"No denoised LAS files found in {folders['denoised_dir']}")
		return

	source_prefix = str(folders["denoised_dir"]) + os.sep
	output_prefix = str(folders["ortho_dir"]) + os.sep
	platform_name = _platform_flag()

	created_count = 0
	skipped_count = 0

	for las_file in denoised_files:
		stem = las_file.stem
		expected_output = folders["ortho_dir"] / f"{stem}RGB.jpg"

		if expected_output.exists() and not args.overwrite:
			print(f"Skipping existing orthoimage: {expected_output.name}")
			skipped_count += 1
			continue

		main2(
			glb_file_path=source_prefix,
			pointName=stem,
			downsample=args.downsample,
			GSDmm2px=args.gsd_mm_per_px,
			bool_alignOnly=False,
			b=platform_name,
			bool_generate=False,
			output_folder_path=output_prefix,
		)
		created_count += 1

	print(
		"Finished create_orthoimages. "
		f"New files created: {created_count}, skipped existing: {skipped_count}"
	)


def cmd_create_dataset(args):
	folders = ensure_project_folders(args.workspace_root)
	from create_dataset import create_dataset

	summary = create_dataset(
		orthoimage_dir=str(folders["ortho_dir"]),
		dataset_output_root=str(folders["dataset_dir"]),
		validation_split=args.validation_split,
		target_height=args.target_height,
		patch_size=args.patch_size,
		stride=args.stride,
		white_threshold=args.white_threshold,
		seed=args.seed,
		copy_unmatched_to_test=not args.no_copy_unmatched_to_test,
	)

	print(f"Dataset created at: {summary['dataset_root']}")


def build_parser():
	parser = argparse.ArgumentParser(
		description=(
			"Unified pointcloud tool that converts PLY -> LAS, denoises LAS, "
			"creates orthoimages, and builds U-Net datasets."
		)
	)
	subparsers = parser.add_subparsers(dest="command", required=True)

	convert_parser = subparsers.add_parser(
		"convert_to_las",
		help="Convert all pointcloud/ply/*.ply to pointcloud/las/*.las (skip existing)",
	)
	convert_parser.add_argument("--workspace-root", default=str(DEFAULT_WORKSPACE_ROOT))
	convert_parser.add_argument("--chunk-size", type=int, default=1_000_000)
	convert_parser.set_defaults(func=cmd_convert_to_las)

	denoise_parser = subparsers.add_parser(
		"remove_isolated_points",
		help="Denoise all pointcloud/las/*.las into pointcloud/las/denoised/*.las (skip existing)",
	)
	denoise_parser.add_argument("--workspace-root", default=str(DEFAULT_WORKSPACE_ROOT))
	denoise_parser.add_argument("--voxel-size", type=float, default=0.05)
	denoise_parser.add_argument("--radius", type=float, default=0.15)
	denoise_parser.add_argument("--min-neighbors", type=int, default=8)
	denoise_parser.add_argument("--chunk-size", type=int, default=1_000_000)
	denoise_parser.set_defaults(func=cmd_remove_isolated_points)

	ortho_parser = subparsers.add_parser(
		"create_orthoimages",
		help="Create orthoimages from pointcloud/las/denoised/*.las into images/orthoimages",
	)
	ortho_parser.add_argument("--workspace-root", default=str(DEFAULT_WORKSPACE_ROOT))
	ortho_parser.add_argument("--downsample", type=int, default=10)
	ortho_parser.add_argument("--gsd-mm-per-px", type=float, default=5.0)
	ortho_parser.add_argument("--overwrite", action="store_true")
	ortho_parser.set_defaults(func=cmd_create_orthoimages)

	dataset_parser = subparsers.add_parser(
		"create_dataset",
		help="Build a timestamped membrane dataset from images/orthoimages",
	)
	dataset_parser.add_argument("--workspace-root", default=str(DEFAULT_WORKSPACE_ROOT))
	dataset_parser.add_argument("--validation-split", type=float, default=0.2)
	dataset_parser.add_argument("--target-height", type=int, default=512)
	dataset_parser.add_argument("--patch-size", type=int, default=512)
	dataset_parser.add_argument("--stride", type=int, default=256)
	dataset_parser.add_argument("--white-threshold", type=float, default=0.98)
	dataset_parser.add_argument("--seed", type=int, default=42)
	dataset_parser.add_argument("--no-copy-unmatched-to-test", action="store_true")
	dataset_parser.set_defaults(func=cmd_create_dataset)

	return parser


def main():
	parser = build_parser()
	args = parser.parse_args()
	args.func(args)


if __name__ == "__main__":
	main()
