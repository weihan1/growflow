"""Extract stable vertex trajectories from verified, visibility-varying meshes.

Usage: python -m scripts.extract_gt_trajectories --data-dir PATH --anchor-index 0

Requires meshes/export_manifest.json and meshes/export_vertex_map.json. It does
not infer correspondences from nearest neighbors, merge vertices, reorder time,
or fill absent geometry with invented positions. The legacy extractor is unchanged.
"""

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from helpers.trajectory_io import trajectory_validity


def read_ply_vertices(path):
    """Read the XYZ-only ASCII PLY files produced by this scene's exporter."""
    with Path(path).open() as stream:
        if stream.readline().strip() != "ply" or stream.readline().strip() != "format ascii 1.0":
            raise ValueError(f"Expected ASCII PLY: {path}")
        count = None
        properties = []
        element = None
        for line in stream:
            fields = line.split()
            if fields[:2] == ["element", "vertex"]:
                count = int(fields[2])
            if fields and fields[0] == "element":
                element = fields[1]
            if fields and fields[0] == "property" and element == "vertex":
                properties.append(fields[1:])
            if fields == ["end_header"]:
                break
        else:
            raise ValueError(f"Missing PLY header terminator: {path}")
        if count is None or count <= 0 or properties != [["float", c] for c in "xyz"]:
            raise ValueError(f"Expected nonempty XYZ float vertices: {path}")
        vertices = np.loadtxt(stream, dtype=np.float32, max_rows=count, ndmin=2)
    if vertices.shape != (count, 3) or not np.isfinite(vertices).all():
        raise ValueError(f"Invalid or truncated PLY vertex data: {path}")
    return vertices


def validate_ranges(ranges):
    """Require an exact partition of the PLY rows, with each object listed once."""
    offset = 0
    seen = set()
    for item in ranges:
        name, start, count = item["object"], item["start"], item["count"]
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("Duplicate or invalid object name in vertex map")
        if type(start) is not int or type(count) is not int or start != offset or count <= 0:
            raise ValueError("Vertex map has overlapping, missing, or invalid ranges")
        seen.add(name)
        offset += count
    if not seen:
        raise ValueError("Vertex map contains no objects")
    return offset


def validate_camera_times(data_dir, frame_count):
    reference = None
    for split in ("train", "test"):
        transforms = json.loads((data_dir / f"transforms_{split}.json").read_text())
        cameras = {}
        for frame in transforms["frames"]:
            path = Path(frame["file_path"])
            if len(path.parts) != 3 or path.parts[0] != split or path.suffix != ".png":
                raise ValueError(f"Unexpected image path: {path}")
            index = int(path.stem)
            if path.name != f"{index:05d}.png" or not 0 <= index < frame_count:
                raise ValueError(f"Invalid image timestep: {path}")
            frames = cameras.setdefault(path.parent.name, {})
            if index in frames:
                raise ValueError(f"Duplicate camera frame: {path}")
            frames[index] = float(frame["time"])
        if not cameras:
            raise ValueError(f"No cameras in {split}")
        for camera, frames in cameras.items():
            if set(frames) != set(range(frame_count)):
                raise ValueError(f"Incomplete camera sequence: {split}/{camera}")
            times = np.array([frames[i] for i in range(frame_count)])
            if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                raise ValueError("Camera times must be finite and strictly increasing")
            if reference is None:
                reference = times
            elif not np.allclose(reference, times, rtol=0, atol=1e-8):
                raise ValueError("Camera times disagree across views or splits")
    return reference


def validate_frame_order(indices):
    """Return the physical frame order while preserving image sequence order."""
    differences = np.diff(indices)
    if np.all(differences > 0):
        return "chronological"
    if np.all(differences < 0):
        return "reverse_chronological"
    raise ValueError("Blender frame indices must be strictly increasing or strictly decreasing")


def trajectory_group(data_dir, manifest):
    """Resolve the plant-only mesh group from exporter metadata."""
    scene = manifest.get("scene")
    if scene is None:
        # Compatibility with verified manifests created before the scene field.
        scene = Path(data_dir).name.split("_transparent_final")[0]
    if not isinstance(scene, str) or not scene:
        raise ValueError("Export manifest has an invalid scene name")
    group = f"relevant_{scene}_meshes"
    if not all(group in frame for frame in manifest.get("frames", [])):
        raise ValueError(f"Export manifest is missing plant-only group: {group}")
    return group


def build_trajectories(data_dir, anchor_index=0):
    data_dir = Path(data_dir)
    mesh_root = data_dir / "meshes"
    manifest = json.loads((mesh_root / "export_manifest.json").read_text())
    group = trajectory_group(data_dir, manifest)
    vertex_map = json.loads((mesh_root / "export_vertex_map.json").read_text())
    frames = sorted(manifest["frames"], key=lambda frame: frame["image_index"])
    if not frames or [f["image_index"] for f in frames] != list(range(len(frames))):
        raise ValueError("Manifest image indices must cover consecutive timesteps")
    if not 0 <= anchor_index < len(frames):
        raise ValueError("Anchor index is outside the sequence")
    indices = np.load(data_dir / "unique_mesh_indices.npy", allow_pickle=False)
    expected_indices = np.array([f["blender_frame"] for f in frames])
    if not np.array_equal(indices, expected_indices):
        raise ValueError("Blender frame indices must agree with the manifest image order")
    frame_order = validate_frame_order(indices)
    manifest_order = manifest.get("frame_order")
    if manifest_order is not None and manifest_order != frame_order:
        raise ValueError("Manifest frame order disagrees with Blender frame indices")
    times = validate_camera_times(data_dir, len(frames))
    maps = {f["image_name"]: f for f in vertex_map["frames"]}
    if len(maps) != len(vertex_map["frames"]) or set(maps) != {f["image_name"] for f in frames}:
        raise ValueError("Vertex map frames do not match the export manifest")

    objects = {}
    total_points = 0
    for frame in frames:
        index = frame["image_index"]
        if frame["image_name"] != f"{index:05d}.png" or frame["mesh_name"] != f"mesh_{frame['blender_frame']:04d}.ply":
            raise ValueError("Incorrect image-to-mesh filename mapping")
        mapping = maps[frame["image_name"]]
        if mapping["blender_frame"] != frame["blender_frame"]:
            raise ValueError("Vertex map Blender frame disagrees with manifest")
        ranges = mapping[group]
        validate_ranges(ranges)
        if set(frame[group]) != {item["object"] for item in ranges}:
            raise ValueError("Selected objects disagree with manifest")
        for item in ranges:
            name, count = item["object"], item["count"]
            if name not in objects:
                objects[name] = {"name": name, "start": total_points, "count": count}
                total_points += count
            elif objects[name]["count"] != count:
                raise ValueError(f"Vertex count changes within object: {name}")

    points = np.full((len(frames), total_points, 3), np.nan, dtype=np.float32)
    validity = np.zeros(points.shape[:2], dtype=bool)
    for t, frame in enumerate(frames):
        ranges = maps[frame["image_name"]][group]
        vertices = read_ply_vertices(mesh_root / group / frame["mesh_name"])
        if len(vertices) != validate_ranges(ranges):
            raise ValueError(f"Vertex map length differs from mesh: {frame['mesh_name']}")
        for item in ranges:
            start, count = item["start"], item["count"]
            target = objects[item["object"]]["start"]
            points[t, target:target + count] = vertices[start:start + count]
            validity[t, target:target + count] = True
        if not np.array_equal(trajectory_validity(points[t]), validity[t]):
            raise ValueError("Trajectory coordinates and validity mask disagree")
        print(f"{frame['image_name']}: {validity[t].sum()} valid / {total_points} track IDs", flush=True)

    metadata = {
        "format_version": 1,
        "point_identity": "object name + local evaluated-mesh vertex index",
        "coordinate_system": manifest["coordinate_system"],
        "frame_order": frame_order,
        "sequence_semantics": "Frames remain in image/training-time order; no extraction-time reversal or anchor reordering.",
        "missing_points": "All three coordinates are NaN when validity is false; never interpolate absent geometry.",
        "validity_semantics": "Object is render-enabled and present in the exported mesh; not per-camera occlusion visibility.",
        "anchor_index": anchor_index,
        "point_count": total_points,
        "objects": list(objects.values()),
        "frames": [dict(image_index=f["image_index"], image_name=f["image_name"],
                        blender_frame=int(f["blender_frame"]), mesh_name=f["mesh_name"],
                        time=float(times[t]), valid_points=int(validity[t].sum()))
                   for t, f in enumerate(frames)],
    }
    return points, validity, metadata


def write_trajectories(output_dir, points, validity, metadata):
    output_dir = Path(output_dir)
    names = ("trajectory_frames.npz", "trajectory_visibility.npz", "trajectory_metadata.json")
    if any((output_dir / name).exists() for name in names):
        raise FileExistsError("Trajectory output already exists; refusing to overwrite it")
    if points.ndim != 3 or validity.shape != points.shape[:2]:
        raise ValueError("Invalid trajectory/visibility shapes")
    for frame, mask in zip(points, validity):
        if not np.array_equal(trajectory_validity(frame), mask):
            raise ValueError("Trajectory coordinates and validity mask disagree")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Publish only fully written, round-trip-checked archives. Hard links avoid
    # overwriting outputs even if another extraction starts concurrently.
    with tempfile.TemporaryDirectory(prefix=".trajectory-", dir=output_dir) as staging:
        staging = Path(staging)
        payload = {f"frame_{i:04d}": frame for i, frame in enumerate(points)}
        masks = {f"frame_{i:04d}": mask for i, mask in enumerate(validity)}
        np.savez_compressed(staging / names[0], **payload)
        np.savez_compressed(staging / names[1], **masks)
        (staging / names[2]).write_text(json.dumps(metadata, indent=2) + "\n")
        for name, expected in ((names[0], payload), (names[1], masks)):
            with np.load(staging / name, allow_pickle=False) as archive:
                if set(archive.files) != set(expected):
                    raise ValueError("Written archive keys do not match frames")
                for key, value in expected.items():
                    if not np.array_equal(archive[key], value, equal_nan=True):
                        raise ValueError("Written archive failed round-trip verification")
        published = []
        try:
            for name in names:
                os.link(staging / name, output_dir / name)
                published.append(output_dir / name)
        except Exception:
            for path in published:
                path.unlink()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--anchor-index", type=int, default=0)
    args = parser.parse_args()
    manifest = json.loads((args.data_dir / "meshes" / "export_manifest.json").read_text())
    output_dir = args.data_dir / "meshes" / trajectory_group(args.data_dir, manifest)
    if any((output_dir / name).exists() for name in
           ("trajectory_frames.npz", "trajectory_visibility.npz", "trajectory_metadata.json")):
        parser.error("Trajectory outputs already exist; refusing to overwrite")
    points, validity, metadata = build_trajectories(args.data_dir, args.anchor_index)
    write_trajectories(output_dir, points, validity, metadata)
    print(f"Saved {len(points)} frames with {points.shape[1]} stable track IDs to {output_dir}")


if __name__ == "__main__":
    main()
