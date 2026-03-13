import os

import laspy
import numpy as np
from plyfile import PlyData


def _resolve_color_fields(vertex_dtype_names):
    names = set(vertex_dtype_names)
    if {"red", "green", "blue"}.issubset(names):
        return "red", "green", "blue"
    if {"r", "g", "b"}.issubset(names):
        return "r", "g", "b"
    return None


def _to_las_color(channel_values):
    if channel_values.dtype == np.uint16:
        return channel_values

    if np.issubdtype(channel_values.dtype, np.floating):
        max_val = float(np.max(channel_values)) if channel_values.size else 0.0
        if max_val <= 1.0:
            scaled = np.clip(channel_values, 0.0, 1.0) * 65535.0
        else:
            scaled = np.clip(channel_values, 0.0, 255.0) / 255.0 * 65535.0
        return scaled.astype(np.uint16)

    return (channel_values.astype(np.uint16) * 257)

def convert_to_las(input_dir, output_dir=None, chunk_size=1_000_000, skip_existing=True):
    """Convert .ply files from ``input_dir`` into .las files in ``output_dir``.

    ``output_dir`` defaults to ``input_dir`` for backward compatibility.
    Returns a list of newly generated LAS file paths.
    """
    las_files_created = []
    if output_dir is None:
        output_dir = input_dir

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")

    for file_name in sorted(os.listdir(input_dir)):
        file_path = os.path.join(input_dir, file_name)

        if not os.path.isfile(file_path):
            continue

        base, ext = os.path.splitext(file_name)
        ext = ext.lower()

        if ext != ".ply":
            continue

        las_path = os.path.join(output_dir, f"{base}.las")

        if skip_existing and os.path.exists(las_path):
            print(f"Skipping existing conversion: {os.path.basename(las_path)}")
            continue

        try:
            ply = PlyData.read(file_path, mmap=True)

            if "vertex" not in ply:
                print(f"No vertex element found in {file_name}, skipping.")
                continue

            vertices = ply["vertex"].data
            total_points = len(vertices)

            if total_points == 0:
                print(f"No points found in {file_name}, skipping.")
                continue

            required_xyz = {"x", "y", "z"}
            if not required_xyz.issubset(set(vertices.dtype.names)):
                raise ValueError(f"PLY file {file_name} is missing one of required fields: x, y, z")

            color_fields = _resolve_color_fields(vertices.dtype.names)

            header = laspy.LasHeader(point_format=2, version="1.2")
            header.scales = np.array([0.001, 0.001, 0.001])

            with laspy.open(las_path, mode="w", header=header) as writer:
                for start in range(0, total_points, chunk_size):
                    end = min(start + chunk_size, total_points)
                    chunk = vertices[start:end]

                    record = laspy.ScaleAwarePointRecord.zeros(len(chunk), header=header)
                    record.x = chunk["x"]
                    record.y = chunk["y"]
                    record.z = chunk["z"]

                    if color_fields is not None:
                        r_name, g_name, b_name = color_fields
                        record.red = _to_las_color(chunk[r_name])
                        record.green = _to_las_color(chunk[g_name])
                        record.blue = _to_las_color(chunk[b_name])

                    writer.write_points(record)

                print(f"Converted PLY -> LAS: {las_path}")
            las_files_created.append(las_path)
        except Exception as exc:
            print(f"Error converting {file_name}: {exc}")

            print(f"PLY conversion complete. New LAS files: {len(las_files_created)}")
    return las_files_created