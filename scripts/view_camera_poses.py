"""Inspect dataset images, camera poses, and mature geometry in a Viser viewer.

Examples:
    python -m scripts.view_camera_poses synthetic --data-dir data/synthetic/clematis_transparent_final_small_vase_70_timesteps_subsample_6
    python -m scripts.view_camera_poses captured --data-dir data/captured/pi_paperwhite_full_subset4
    python -m scripts.view_camera_poses captured --data-dir data/captured/pi_paperwhite_full_subset4 --timestep 3

Synthetic poses are deduplicated across time by camera name. The first (mature)
image for each camera is shown in the side panel when clicked. Captured scenes use the latest
COLMAP point cloud because they do not include a ground-truth mesh.
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import numpy as np


@dataclass
class CameraPose:
    name: str
    split: str
    c2w: np.ndarray
    fov_y: float
    aspect: float
    image_path: Path
    # Captured-camera calibration. Synthetic images are already pinhole renders.
    calibration: tuple | None = None


def _rigid_rotation(matrix: np.ndarray) -> np.ndarray:
    """Remove scale from a camera matrix (Blender exports can contain scale)."""
    u, _, vh = np.linalg.svd(matrix[:3, :3])
    correction = np.diag([1.0, 1.0, np.linalg.det(u @ vh)])
    return u @ correction @ vh


def load_synthetic(data_dir: Path, split: str) -> list[CameraPose]:
    from PIL import Image

    poses = []
    for current_split in ("train", "test") if split == "both" else (split,):
        transforms_path = data_dir / f"transforms_{current_split}.json"
        with transforms_path.open() as handle:
            transforms = json.load(handle)
        frames = transforms["frames"]
        if not frames:
            continue

        # Image.open reads the header only; no full image is decoded.
        first_image = data_dir / frames[0]["file_path"].removeprefix("./")
        with Image.open(first_image) as image:
            width, height = image.size
        aspect = width / height
        fov_x = float(transforms["camera_angle_x"])
        fov_y = 2.0 * np.arctan(np.tan(fov_x / 2.0) / aspect)

        seen = {}
        for frame in frames:
            path = Path(frame["file_path"])
            name = path.parent.name
            c2w = np.asarray(frame["transform_matrix"], dtype=np.float64).copy()
            if c2w.shape != (4, 4):
                raise ValueError(f"Invalid transform for {frame['file_path']}")
            c2w[:3, 1:3] *= -1  # Blender/OpenGL to the OpenCV convention used by training and Viser.
            if name in seen:
                if not np.allclose(c2w, seen[name], atol=1e-5):
                    raise ValueError(f"Camera {name} has changing poses across time")
                continue
            seen[name] = c2w
            poses.append(CameraPose(name, current_split, c2w, float(fov_y), aspect,
                                    data_dir / frame["file_path"].removeprefix("./")))
    return poses


def load_captured(data_dir: Path, timestep: int, test_every: int) -> tuple[list[CameraPose], str]:
    from pycolmap import SceneManager

    dates = sorted((p for p in data_dir.glob("timelapse*") if p.is_dir()), reverse=True)
    if not dates:
        raise ValueError(f"No timelapse directories found in {data_dir}")
    if timestep < 0 or timestep >= len(dates):
        raise ValueError(f"Timestep must be between 0 and {len(dates) - 1}")
    date_dir = dates[timestep]
    sparse_dir = date_dir / "sparse" / "0"
    if not sparse_dir.is_dir():
        sparse_dir = date_dir / "sparse"
    manager = SceneManager(str(sparse_dir))
    manager.load_cameras()
    manager.load_images()

    # DynamicParser sorts image names before assigning train/test indices.
    poses = []
    for index, image in enumerate(sorted(manager.images.values(), key=lambda item: item.name)):
        camera = manager.cameras[image.camera_id]
        w2c = np.eye(4)
        w2c[:3, :3] = image.R()
        w2c[:3, 3] = image.tvec
        c2w = np.linalg.inv(w2c)
        fov_y = 2.0 * np.arctan(camera.height / (2.0 * camera.fy))
        split = "test" if index % test_every == 0 else "train"
        calibration = (camera.width, camera.height, camera.fx, camera.fy,
                       camera.cx, camera.cy, camera.camera_type,
                       tuple(getattr(camera, key, 0.0) for key in ("k1", "k2", "p1", "p2")))
        poses.append(CameraPose(image.name, split, c2w, float(fov_y),
                                camera.width / camera.height,
                                date_dir / "images_still" / image.name, calibration))
    return poses, date_dir.name


def load_thumbnail(pose: CameraPose, max_width: int = 320) -> tuple[np.ndarray, float, float]:
    """Return a small RGB image and its matching centered-pinhole frustum parameters."""
    import cv2
    from PIL import Image

    with Image.open(pose.image_path) as source:
        if source.mode == "RGBA":
            rgba = np.asarray(source, dtype=np.float32) / 255.0
            image = (rgba[..., :3] * rgba[..., 3:4] * 255).astype(np.uint8)
        else:
            image = np.asarray(source.convert("RGB"))

    if pose.calibration is None:
        fov_y, aspect = pose.fov_y, pose.aspect
    else:
        width, height, fx, fy, cx, cy, camera_type, (k1, k2, p1, p2) = pose.calibration
        if image.shape[:2] != (height, width):
            sx, sy = image.shape[1] / width, image.shape[0] / height
            fx, fy, cx, cy = fx * sx, fy * sy, cx * sx, cy * sy
            width, height = image.shape[1], image.shape[0]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        if camera_type in (2, 3, 4):
            distortion = np.array([k1, k2, p1, p2], dtype=np.float64)
            new_K, (x, y, w, h) = cv2.getOptimalNewCameraMatrix(
                K, distortion, (width, height), 0
            )
            image = cv2.undistort(image, K, distortion, None, new_K)[y:y + h, x:x + w]
            fx, fy, cx, cy = new_K[0, 0], new_K[1, 1], new_K[0, 2] - x, new_K[1, 2] - y
        elif camera_type not in (0, 1):
            raise ValueError(f"Unsupported COLMAP camera model {camera_type} for {pose.name}")

        # Viser uses a centered principal point. Pad the image so the COLMAP
        # principal point becomes central, without changing the projection.
        height, width = image.shape[:2]
        new_width = int(np.ceil(2 * max(cx, width - cx)))
        new_height = int(np.ceil(2 * max(cy, height - cy)))
        left = int(round(new_width / 2 - cx))
        top = int(round(new_height / 2 - cy))
        image = cv2.copyMakeBorder(image, top, new_height - height - top,
                                   left, new_width - width - left,
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))
        fov_y = float(2 * np.arctan(new_height / (2 * fy)))
        aspect = new_width / new_height

    if image.shape[1] > max_width:
        target = (max_width, max(1, round(image.shape[0] * max_width / image.shape[1])))
        image = cv2.resize(image, target, interpolation=cv2.INTER_AREA)
    return image, fov_y, aspect


def add_mature_geometry(server, data_dir: Path, dataset: str,
                        date: str | None = None, mesh_path: Path | None = None) -> str:
    if mesh_path is not None or dataset == "synthetic":
        import trimesh

        if mesh_path is None:
            meshes = sorted((data_dir / "meshes" / "relevant_meshes").glob("mesh_*.ply"))
            if not meshes:
                raise FileNotFoundError(f"No mature mesh found in {data_dir / 'meshes' / 'relevant_meshes'}")
            mesh_path = meshes[-1]
        mature_mesh = mesh_path
        mesh = trimesh.load(mature_mesh, force="mesh", process=False)
        server.scene.add_mesh_trimesh("/mature_mesh", mesh)
        return str(mature_mesh)

    from pycolmap import SceneManager

    sparse_dir = data_dir / date / "sparse" / "0"
    if not sparse_dir.is_dir():
        sparse_dir = data_dir / date / "sparse"
    manager = SceneManager(str(sparse_dir))
    manager.load_points3D()
    server.scene.add_point_cloud("/mature_colmap_points", manager.points3D.astype(np.float32),
                                 manager.point3D_colors.astype(np.uint8), point_size=0.012)
    return str(sparse_dir / "points3D.bin")


def discover_scenes(data_dir: Path, dataset: str) -> tuple[dict[str, Path], str]:
    """Accept either one scene or a dataset root and list switchable scenes."""
    def is_scene(path: Path) -> bool:
        if dataset == "synthetic":
            return (path / "transforms_train.json").is_file()
        return any(child.is_dir() for child in path.glob("timelapse*"))

    selected_path = data_dir if is_scene(data_dir) else None
    root = data_dir.parent if selected_path is not None else data_dir
    scenes = {}
    selected = None
    for path in sorted(root.iterdir()):
        if not path.is_dir() or not is_scene(path):
            continue
        if dataset == "synthetic":
            plant = path.name.split("_transparent_final", 1)[0]
            variant = path.name.rsplit("_subsample_", 1)
            label = f"{plant} (subsample {variant[1]})" if len(variant) == 2 else f"{plant} (full)"
        else:
            label = path.name.removeprefix("pi_").replace("_", " ")
        if label in scenes:
            label = path.name  # Retain both entries if shortened names collide.
        scenes[label] = path
        if path == selected_path:
            selected = label
    if not scenes:
        raise ValueError(f"No {dataset} scenes found in {root}")
    return scenes, selected or next(iter(scenes))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("synthetic", "captured"))
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Scene directory or parent dataset directory")
    parser.add_argument("--split", choices=("train", "test", "both"), default="both",
                        help="Synthetic split to show (default: both)")
    parser.add_argument("--timestep", type=int, default=0,
                        help="Captured timestep; 0 is the latest date (default: 0)")
    parser.add_argument("--test-every", type=int, default=8,
                        help="Captured test camera interval, matching DynamicParser (default: 8)")
    parser.add_argument("--mesh", type=Path,
                        help="Optional mesh already aligned with the dataset's camera coordinates")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    if args.test_every < 1:
        parser.error("--test-every must be positive")
    if not args.data_dir.is_dir():
        parser.error(f"Dataset directory not found: {args.data_dir}")
    if args.mesh is not None and not args.mesh.is_file():
        parser.error(f"Mesh file not found: {args.mesh}")
    try:
        scenes, initial_scene = discover_scenes(args.data_dir, args.dataset)
    except ValueError as error:
        parser.error(str(error))

    import viser
    from viser import transforms as tf

    server = viser.ViserServer(host=args.host, port=args.port, label="Camera pose viewer")
    dropdown = server.gui.add_dropdown("Plant", options=tuple(scenes), initial_value=initial_scene)
    status = server.gui.add_markdown("Loading scene...")
    colors = {"train": (55, 160, 255), "test": (255, 130, 70)}
    client_panels = {}
    state = {"scene": None, "poses": [], "thumbnails": {}, "extent": 1.0}
    switch_lock = RLock()

    def set_overview(client):
        pose = state["poses"][0]
        rotation = _rigid_rotation(pose.c2w)
        forward = rotation[:, 2]
        target = pose.c2w[:3, 3] + forward * max(state["extent"], 0.5)
        client.camera.position = target - forward * max(3 * state["extent"], 1.5)
        client.camera.look_at = target
        client.camera.up_direction = -rotation[:, 1]
        client.camera.fov = 1.0

    def show_camera(client, pose, rotation):
        key = (pose.split, pose.name)
        with switch_lock:
            if key not in state["thumbnails"]:
                state["thumbnails"][key] = load_thumbnail(pose)
            thumbnail, fov_y, _ = state["thumbnails"][key]
            camera = client.camera
            camera.position = pose.c2w[:3, 3]
            camera.wxyz = tf.SO3.from_matrix(rotation).wxyz
            camera.fov = fov_y
            panel = client_panels.get(client.client_id)
            if panel is not None:
                label, preview = panel
                label.content = f"**{state['scene']} / {pose.split} / {pose.name}**"
                preview.image = thumbnail

    def switch_scene(scene_name: str) -> None:
        with switch_lock:
            data_dir = scenes[scene_name]
            status.content = f"Loading **{scene_name}**..."
            if args.dataset == "synthetic":
                poses = load_synthetic(data_dir, args.split)
                date = None
            else:
                poses, date = load_captured(data_dir, args.timestep, args.test_every)
            if not poses:
                raise ValueError(f"No camera poses found in {data_dir}")

            first_thumbnail = load_thumbnail(poses[0])
            centers = np.stack([pose.c2w[:3, 3] for pose in poses])
            center = centers.mean(axis=0)
            extent = np.max(np.linalg.norm(centers - center, axis=1))
            scale = max(0.05, float(extent) * 0.12)
            server.scene.reset()
            server.scene.add_frame("/world", axes_length=scale * 2)
            # A user-supplied mesh belongs to the initially selected scene only.
            mesh_override = args.mesh if scene_name == initial_scene else None
            geometry_path = add_mature_geometry(server, data_dir, args.dataset, date, mesh_override)
            state.update(scene=scene_name, poses=poses, extent=float(extent),
                         thumbnails={(poses[0].split, poses[0].name): first_thumbnail})

            for pose in poses:
                rotation = _rigid_rotation(pose.c2w)
                frustum = server.scene.add_camera_frustum(
                    f"/{pose.split}/{pose.name}",
                    fov=pose.fov_y,
                    aspect=pose.aspect,
                    scale=scale,
                    color=colors[pose.split],
                    position=pose.c2w[:3, 3],
                    wxyz=tf.SO3.from_matrix(rotation).wxyz,
                )

                @frustum.on_click
                def _on_camera_click(event, pose=pose, rotation=rotation, scene_name=scene_name):
                    if state["scene"] == scene_name:
                        show_camera(event.client, pose, rotation)

            for label, preview in client_panels.values():
                label.content = f"**{scene_name}** — click a camera frustum to inspect its view."
                preview.image = first_thumbnail[0]
            for client in server.get_clients().values():
                set_overview(client)
            train_count = sum(pose.split == "train" for pose in poses)
            status.content = (f"**{scene_name}**: {train_count} train (blue), "
                              f"{len(poses) - train_count} test (orange) cameras")
            print(f"{scene_name}: {len(poses)} cameras; geometry: {geometry_path}")

    @dropdown.on_update
    def _on_plant_change(_):
        if dropdown.value == state["scene"]:
            return
        try:
            switch_scene(dropdown.value)
        except Exception as error:
            status.content = f"Could not load **{dropdown.value}**: {error}"
            print(f"Scene switch failed: {error}")
            dropdown.value = state["scene"]

    @server.on_client_connect
    def _on_client_connect(client):
        with switch_lock:
            if state["poses"]:
                first_pose = state["poses"][0]
                first_thumbnail = state["thumbnails"][(first_pose.split, first_pose.name)][0]
            else:
                first_thumbnail = np.zeros((1, 1, 3), dtype=np.uint8)
            label = client.gui.add_markdown("Click a camera frustum to inspect its view.")
            preview = client.gui.add_image(first_thumbnail, label="Camera image")
            client_panels[client.client_id] = (label, preview)
            if state["poses"]:
                set_overview(client)

    @server.on_client_disconnect
    def _on_client_disconnect(client):
        with switch_lock:
            client_panels.pop(client.client_id, None)

    switch_scene(initial_scene)
    print(f"Open http://{args.host}:{args.port}  (Ctrl-C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
