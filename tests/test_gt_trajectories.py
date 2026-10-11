import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from helpers.trajectory_io import select_trajectory_points, trajectory_validity
from scripts.extract_gt_trajectories import build_trajectories, write_trajectories


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "testplant_transparent_final_3_timesteps"
        self.mesh_root = self.root / "meshes"
        self.group = "relevant_testplant_meshes"
        self.output = self.mesh_root / self.group
        self.output.mkdir(parents=True)
        # A disappears and returns; B moves ahead of A in the PLY ordering;
        # C is a different organ that first appears at the last frame.
        self.objects = [
            [("A", [[1, 2, 3], [4, 5, 6]]), ("B", [[10, 20, 30]])],
            [("B", [[11, 21, 31]])],
            [("C", [[100, 200, 300]]), ("B", [[12, 22, 32]]), ("A", [[3, 4, 5], [6, 7, 8]])],
        ]
        self.blender_frames = [60, 182, 320]
        manifest_frames = []
        map_frames = []
        for i, (frame, objects) in enumerate(zip(self.blender_frames, self.objects)):
            name = f"mesh_{frame:04d}.ply"
            vertices = np.concatenate([np.asarray(points) for _, points in objects])
            header = ("ply\nformat ascii 1.0\n" + f"element vertex {len(vertices)}\n"
                      + "property float x\nproperty float y\nproperty float z\n"
                      + "element face 0\nproperty list uchar uint vertex_indices\nend_header\n")
            body = "\n".join(" ".join(map(str, row)) for row in vertices) + "\n"
            (self.output / name).write_text(header + body)
            ranges = []
            offset = 0
            for object_name, points in objects:
                ranges.append({"object": object_name, "start": offset, "count": len(points)})
                offset += len(points)
            manifest_frames.append({"image_index": i, "image_name": f"{i:05d}.png",
                                    "blender_frame": frame, "mesh_name": name,
                                    self.group: [n for n, _ in objects]})
            map_frames.append({"image_name": f"{i:05d}.png", "blender_frame": frame,
                               self.group: ranges})
        self.manifest = {"scene": "testplant", "coordinate_system": "Blender world",
                         "frame_order": "chronological", "frames": manifest_frames}
        self.vertex_map = {"frames": map_frames}
        self.write_json("export_manifest.json", self.manifest)
        self.write_json("export_vertex_map.json", self.vertex_map)
        np.save(self.root / "unique_mesh_indices.npy", np.array(self.blender_frames, dtype=np.int32))
        for split in ("train", "test"):
            frames = [{"file_path": f"./{split}/r_0/{i:05d}.png", "time": i / 2}
                      for i in range(3)]
            (self.root / f"transforms_{split}.json").write_text(json.dumps({"frames": frames}))

    def write_json(self, name, value):
        (self.mesh_root / name).write_text(json.dumps(value))

    def test_identity_survives_reordering_disappearance_and_reappearance(self):
        points, valid, meta = build_trajectories(self.root, anchor_index=1)
        self.assertEqual(points.shape, (3, 4, 3))
        self.assertEqual([o["name"] for o in meta["objects"]], ["A", "B", "C"])
        np.testing.assert_array_equal(points[:, 2], [[10, 20, 30], [11, 21, 31], [12, 22, 32]])
        np.testing.assert_array_equal(points[2, :2], [[3, 4, 5], [6, 7, 8]])
        self.assertTrue(np.isnan(points[1, :2]).all())
        self.assertFalse(valid[:2, 3].any())
        np.testing.assert_array_equal(points[2, 3], [100, 200, 300])
        self.assertEqual(meta["anchor_index"], 1)
        self.assertEqual(meta["frame_order"], "chronological")
        self.assertEqual([f["blender_frame"] for f in meta["frames"]], [60, 182, 320])

    def test_reverse_chronology_preserves_image_order(self):
        manifest = copy.deepcopy(self.manifest)
        mapping = copy.deepcopy(self.vertex_map)
        manifest["frames"].reverse()
        mapping["frames"].reverse()
        manifest["frame_order"] = "reverse_chronological"
        for i, (frame, mapped) in enumerate(zip(manifest["frames"], mapping["frames"])):
            frame["image_index"] = i
            frame["image_name"] = f"{i:05d}.png"
            mapped["image_name"] = f"{i:05d}.png"
        self.write_json("export_manifest.json", manifest)
        self.write_json("export_vertex_map.json", mapping)
        np.save(self.root / "unique_mesh_indices.npy", np.array(self.blender_frames[::-1]))
        points, valid, meta = build_trajectories(self.root, anchor_index=0)
        self.assertEqual(meta["frame_order"], "reverse_chronological")
        self.assertEqual([f["blender_frame"] for f in meta["frames"]], [320, 182, 60])
        self.assertEqual([o["name"] for o in meta["objects"]], ["C", "B", "A"])
        np.testing.assert_array_equal(points[0, 0], [100, 200, 300])
        np.testing.assert_array_equal(valid.sum(axis=1), [4, 1, 3])

    def test_archives_round_trip_and_refuse_overwrite(self):
        points, valid, meta = build_trajectories(self.root, 1)
        write_trajectories(self.output, points, valid, meta)
        with np.load(self.output / "trajectory_frames.npz", allow_pickle=False) as archive:
            self.assertEqual(archive.files, ["frame_0000", "frame_0001", "frame_0002"])
            for i in range(3):
                np.testing.assert_array_equal(archive[f"frame_{i:04d}"], points[i])
        with np.load(self.output / "trajectory_visibility.npz", allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["frame_0001"], valid[1])
        before = (self.output / "trajectory_frames.npz").read_bytes()
        with self.assertRaises(FileExistsError):
            write_trajectories(self.output, points, valid, meta)
        self.assertEqual(before, (self.output / "trajectory_frames.npz").read_bytes())
        self.assertFalse(list(self.output.glob(".trajectory-*")))

    def test_rejects_overlap_or_gap_in_vertex_map(self):
        for start in (1, 3):
            mapping = copy.deepcopy(self.vertex_map)
            mapping["frames"][0][self.group][1]["start"] = start
            self.write_json("export_vertex_map.json", mapping)
            with self.assertRaisesRegex(ValueError, "ranges"):
                build_trajectories(self.root)

    def test_rejects_changed_object_vertex_count(self):
        mapping = copy.deepcopy(self.vertex_map)
        mapping["frames"][1][self.group][0]["count"] = 2
        self.write_json("export_vertex_map.json", mapping)
        with self.assertRaisesRegex(ValueError, "Vertex count changes"):
            build_trajectories(self.root)

    def test_rejects_mesh_map_length_mismatch(self):
        mapping = copy.deepcopy(self.vertex_map)
        mapping["frames"][2][self.group][0]["count"] = 2
        for item in mapping["frames"][2][self.group][1:]:
            item["start"] += 1
        self.write_json("export_vertex_map.json", mapping)
        with self.assertRaisesRegex(ValueError, "length differs"):
            build_trajectories(self.root)

    def test_rejects_wrong_frame_order_or_mapping(self):
        np.save(self.root / "unique_mesh_indices.npy", np.array(self.blender_frames[::-1]))
        with self.assertRaisesRegex(ValueError, "manifest image order"):
            build_trajectories(self.root)

    def test_rejects_scrambled_or_duplicate_physical_frames(self):
        for indices in ([60, 320, 182], [60, 60, 320]):
            manifest = copy.deepcopy(self.manifest)
            for frame, blender_frame in zip(manifest["frames"], indices):
                frame["blender_frame"] = blender_frame
                frame["mesh_name"] = f"mesh_{blender_frame:04d}.ply"
            manifest.pop("frame_order")
            self.write_json("export_manifest.json", manifest)
            np.save(self.root / "unique_mesh_indices.npy", np.array(indices))
            with self.assertRaisesRegex(ValueError, "strictly"):
                build_trajectories(self.root)

    def test_rejects_duplicate_map_frames(self):
        mapping = copy.deepcopy(self.vertex_map)
        mapping["frames"].append(mapping["frames"][0])
        self.write_json("export_vertex_map.json", mapping)
        with self.assertRaisesRegex(ValueError, "Vertex map frames"):
            build_trajectories(self.root)

    def test_rejects_inconsistent_camera_time(self):
        path = self.root / "transforms_test.json"
        data = json.loads(path.read_text())
        data["frames"][1]["time"] = 0.6
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "times disagree"):
            build_trajectories(self.root)

    def test_rejects_duplicate_camera_frame(self):
        path = self.root / "transforms_train.json"
        data = json.loads(path.read_text())
        data["frames"].append(data["frames"][0])
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "Duplicate camera"):
            build_trajectories(self.root)

    def test_missing_points_are_filtered_without_renumbering_ids(self):
        points = np.array([[1., 2., 3.], [np.nan] * 3, [4., 5., 6.]])
        selected, ids = select_trajectory_points(points, np.array([2, 1, 0]))
        np.testing.assert_array_equal(ids, [2, 0])
        np.testing.assert_array_equal(selected, points[[2, 0]])
        for bad in (np.array([-1]), np.array([3]), np.array([0.5])):
            with self.assertRaises(ValueError):
                select_trajectory_points(points, bad)

    def test_legacy_finite_frames_are_unchanged(self):
        points = np.arange(12, dtype=np.float32).reshape(4, 3)
        selected, ids = select_trajectory_points(points)
        np.testing.assert_array_equal(selected, points)
        np.testing.assert_array_equal(ids, np.arange(4))

    def test_partial_nan_and_infinite_points_fail_loudly(self):
        for points in ([[1, 2, np.nan]], [[np.inf, np.inf, np.inf]]):
            with self.assertRaises(ValueError):
                trajectory_validity(points)


if __name__ == "__main__":
    unittest.main()
