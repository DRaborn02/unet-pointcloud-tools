import argparse
import math
import os
from collections import defaultdict

import laspy
import numpy as np


def _iter_las_files(input_path):
    if os.path.isfile(input_path):
        if not input_path.lower().endswith(".las"):
            raise ValueError("Input file must be a .las file")
        return [input_path]

    if not os.path.isdir(input_path):
        raise ValueError(f"Input path does not exist: {input_path}")

    las_files = [
        os.path.join(input_path, name)
        for name in os.listdir(input_path)
        if name.lower().endswith(".las") and os.path.isfile(os.path.join(input_path, name))
    ]
    return sorted(las_files)


def _make_output_path(input_las_path, output_path, suffix):
    base_name = os.path.basename(input_las_path)
    stem, ext = os.path.splitext(base_name)

    if output_path is None:
        return os.path.join(os.path.dirname(input_las_path), f"{stem}{suffix}{ext}")

    if output_path.lower().endswith(".las"):
        return output_path

    os.makedirs(output_path, exist_ok=True)
    return os.path.join(output_path, f"{stem}{suffix}{ext}")


def _quantize_points(x, y, z, origin, voxel_size):
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    ix = np.floor((x - origin[0]) / voxel_size).astype(np.int64)
    iy = np.floor((y - origin[1]) / voxel_size).astype(np.int64)
    iz = np.floor((z - origin[2]) / voxel_size).astype(np.int64)
    return np.stack((ix, iy, iz), axis=1)


def _accumulate_voxel_counts(las_path, voxel_size, chunk_size):
    voxel_counts = defaultdict(int)
    origin = None
    total_points = 0

    with laspy.open(las_path) as reader:
        for chunk in reader.chunk_iterator(chunk_size):
            x = np.asarray(chunk.x)
            y = np.asarray(chunk.y)
            z = np.asarray(chunk.z)

            if len(x) == 0:
                continue

            if origin is None:
                origin = (float(x[0]), float(y[0]), float(z[0]))

            voxels = _quantize_points(x, y, z, origin, voxel_size)
            unique_voxels, counts = np.unique(voxels, axis=0, return_counts=True)

            for voxel, count in zip(unique_voxels, counts):
                voxel_counts[(int(voxel[0]), int(voxel[1]), int(voxel[2]))] += int(count)

            total_points += len(chunk)

    if origin is None:
        raise ValueError(f"No points found in {las_path}")

    return voxel_counts, origin, total_points


def _build_neighbor_offsets(radius_steps):
    offsets = []
    r2 = radius_steps * radius_steps
    for dx in range(-radius_steps, radius_steps + 1):
        for dy in range(-radius_steps, radius_steps + 1):
            for dz in range(-radius_steps, radius_steps + 1):
                if dx * dx + dy * dy + dz * dz <= r2:
                    offsets.append((dx, dy, dz))
    return offsets


def _find_sparse_voxels(voxel_counts, neighbor_offsets, min_neighbors):
    sparse_voxels = set()

    for vx, vy, vz in voxel_counts.keys():
        local_count = 0
        for dx, dy, dz in neighbor_offsets:
            local_count += voxel_counts.get((vx + dx, vy + dy, vz + dz), 0)
            if local_count >= min_neighbors:
                break

        if local_count < min_neighbors:
            sparse_voxels.add((vx, vy, vz))

    return sparse_voxels


def _write_filtered_las(input_las_path, output_las_path, origin, voxel_size, sparse_voxels, chunk_size):
    kept_points = 0
    removed_points = 0

    with laspy.open(input_las_path) as reader:
        output_header = reader.header.copy()

        with laspy.open(output_las_path, mode="w", header=output_header) as writer:
            for chunk in reader.chunk_iterator(chunk_size):
                if len(chunk) == 0:
                    continue

                voxels = _quantize_points(chunk.x, chunk.y, chunk.z, origin, voxel_size)
                unique_voxels, inverse = np.unique(voxels, axis=0, return_inverse=True)

                voxel_is_sparse = np.array(
                    [
                        (int(v[0]), int(v[1]), int(v[2])) in sparse_voxels
                        for v in unique_voxels
                    ],
                    dtype=bool,
                )

                keep_mask = ~voxel_is_sparse[inverse]

                if np.any(keep_mask):
                    writer.write_points(chunk[keep_mask])

                kept_points += int(np.sum(keep_mask))
                removed_points += int(len(chunk) - np.sum(keep_mask))

    return kept_points, removed_points


def remove_isolated_points(
    input_las_path,
    output_las_path,
    voxel_size=0.005,
    radius=0.005,
    min_neighbors=50,
    chunk_size=1_000_000,
):
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")
    if radius <= 0:
        raise ValueError("radius must be > 0")
    if min_neighbors < 1:
        raise ValueError("min_neighbors must be >= 1")

    radius_steps = max(1, int(math.ceil(radius / voxel_size)))

    print(f"\nScanning: {os.path.basename(input_las_path)}")
    voxel_counts, origin, total_points = _accumulate_voxel_counts(
        input_las_path, voxel_size=voxel_size, chunk_size=chunk_size
    )
    print(f"Total points: {total_points}")
    print(f"Occupied voxels: {len(voxel_counts)}")

    neighbor_offsets = _build_neighbor_offsets(radius_steps)
    sparse_voxels = _find_sparse_voxels(
        voxel_counts=voxel_counts,
        neighbor_offsets=neighbor_offsets,
        min_neighbors=min_neighbors,
    )

    print(f"Sparse voxels to remove: {len(sparse_voxels)}")

    kept_points, removed_points = _write_filtered_las(
        input_las_path=input_las_path,
        output_las_path=output_las_path,
        origin=origin,
        voxel_size=voxel_size,
        sparse_voxels=sparse_voxels,
        chunk_size=chunk_size,
    )

    print(f"Wrote: {output_las_path}")
    print(f"Kept points: {kept_points}")
    print(f"Removed points: {removed_points}")

    return {
        "input": input_las_path,
        "output": output_las_path,
        "total_points": total_points,
        "kept_points": kept_points,
        "removed_points": removed_points,
        "removed_ratio": (removed_points / total_points) if total_points else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Remove isolated point clusters from LAS files using a chunked voxel-density filter."
    )
    parser.add_argument("input", help="Input .las file or folder containing .las files")
    parser.add_argument(
        "--output",
        default=None,
        help="Output .las file (single input) or output directory (folder input). Defaults to *_denoised.las next to each input.",
    )
    parser.add_argument("--suffix", default="_denoised", help="Suffix for output file names")
    parser.add_argument("--voxel-size", type=float, default=0.05, help="Voxel size in meters")
    parser.add_argument("--radius", type=float, default=0.15, help="Neighbor search radius in meters")
    parser.add_argument("--min-neighbors", type=int, default=8, help="Minimum nearby points required to keep a voxel")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Number of points per chunk for LAS streaming",
    )

    args = parser.parse_args()

    las_files = _iter_las_files(args.input)
    if not las_files:
        print("No LAS files found.")
        return

    if len(las_files) > 1 and args.output and args.output.lower().endswith(".las"):
        raise ValueError("When processing a folder, --output must be a directory (or omitted).")

    summaries = []
    for las_path in las_files:
        out_path = _make_output_path(las_path, args.output, args.suffix)
        summary = remove_isolated_points(
            input_las_path=las_path,
            output_las_path=out_path,
            voxel_size=args.voxel_size,
            radius=args.radius,
            min_neighbors=args.min_neighbors,
            chunk_size=args.chunk_size,
        )
        summaries.append(summary)

    removed = sum(item["removed_points"] for item in summaries)
    total = sum(item["total_points"] for item in summaries)
    ratio = (removed / total) if total else 0.0

    print("\n--- Summary ---")
    print(f"Files processed: {len(summaries)}")
    print(f"Total points: {total}")
    print(f"Total removed: {removed}")
    print(f"Removal ratio: {ratio:.4%}")


if __name__ == "__main__":
    main()
