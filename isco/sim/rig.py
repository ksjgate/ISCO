"""Persistent inspection-camera rig (Isaac Sim ``isaacsim.sensors.camera.Camera``).

Conventions
-----------
* Poses come in as ISCO ``T_world_cam`` with **OpenCV axes** (x right, y down, z
  forward). The USD camera prim looks along its local -Z with +Y up, so the prim
  orientation is ``R_usd = R_cv @ diag(1, -1, -1)`` (``opencv_to_usd_rotation``) and the
  position is ``frame.to_stage_pts(t)``; both are applied with
  ``Camera.set_world_pose(position, quat_wxyz, camera_axes="usd")``.
  ``Camera.get_view_matrix_ros()`` (world -> ROS/OpenCV camera) then equals
  ``inv(frame.to_stage_T(T_world_cam))`` (checked in ``tests/isaac``).
* Intrinsics follow ``SensorModel``/``PinholeCamera``: the USD attributes are set
  directly in millimetres (``focalLength`` = 24, ``horizontalAperture`` =
  ``2 f tan(hfov/2)``, ``verticalAperture`` = aperture * H/W -> square pixels). Isaac's
  ``get_intrinsics_matrix()`` gives ``fx = W f / aperture_h``, ``cx = W/2`` - identical to
  ``PinholeCamera.K`` up to float32 rounding (~1e-7 relative).
* Annotators per camera: ``distance_to_image_plane`` (z-depth), ``semantic_segmentation``
  (``colorize=False``, semantic types ``class`` + ``defect_id``) and ``normals``
  (world-space, float4). ``rgb`` / ``pointcloud`` are attached on demand.
* Unused cameras are *parked*: their render product stops updating
  (``hydra_texture.set_updates_enabled(False)``) and the prim is moved far below the
  ground, so they cost (almost) nothing per rendered frame.
"""
from __future__ import annotations

import numpy as np

from ..geometry.camera_model import PinholeCamera
from ..geometry.transforms import look_at_opencv, opencv_to_usd_rotation, quat_wxyz_from_R

DEFAULT_ANNOTATORS = ("distance_to_image_plane", "semantic_segmentation", "normals")
SEMANTIC_TYPES = ["class", "defect_id"]
PARK_POSITION_STAGE = np.array([0.0, 0.0, -100.0])


def look_at_quat_wxyz_gf(eye, target, world_up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """The PoC's original Gf-based ``look_at_quat_wxyz`` (USD camera axes), kept as the
    reference for the Gf-free ``isco.geometry.transforms.look_at_quat_wxyz_usd``."""
    from pxr import Gf

    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(world_up, dtype=float)
    z_axis = eye - target
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rot = np.stack([x_axis, y_axis, z_axis], axis=0)   # rows = camera axes (Gf row-vector convention)
    quat = Gf.Matrix3d(*rot.ravel().tolist()).ExtractRotation().GetQuat()
    imag = quat.GetImaginary()
    return np.array([quat.GetReal(), imag[0], imag[1], imag[2]])


def set_pinhole_intrinsics(camera, pinhole: PinholeCamera, focal_length_mm: float = 24.0,
                           near_m: float = 0.05, far_m: float = 10.0) -> None:
    """Write ``PinholeCamera`` intrinsics onto an Isaac ``Camera`` prim (USD attrs, mm)."""
    f, h_ap, v_ap = pinhole.isaac_apertures(focal_length_mm)
    prim = camera.prim
    prim.GetAttribute("focalLength").Set(float(f))
    prim.GetAttribute("horizontalAperture").Set(float(h_ap))
    prim.GetAttribute("verticalAperture").Set(float(v_ap))
    for name in ("horizontalApertureOffset", "verticalApertureOffset"):
        a = prim.GetAttribute(name)
        if a and a.IsValid():
            a.Set(0.0)
    camera.set_clipping_range(near_m, far_m)


def set_camera_pose_cv(camera, frame, T_world_cam: np.ndarray) -> None:
    """Place an Isaac camera at ISCO ``T_world_cam`` (OpenCV axes)."""
    T_stage = frame.to_stage_T(np.asarray(T_world_cam, dtype=float))
    q = quat_wxyz_from_R(opencv_to_usd_rotation(T_stage[:3, :3]))
    camera.set_world_pose(position=T_stage[:3, 3], orientation=q, camera_axes="usd")


class RigCamera:
    """One persistent camera slot of the rig."""

    def __init__(self, index: int, camera, pinhole: PinholeCamera, annotators):
        self.index = index
        self.camera = camera
        self.pinhole = pinhole
        self.annotators = list(annotators)
        self.active = True
        self.T_world_cam: np.ndarray | None = None

    @property
    def prim_path(self) -> str:
        return self.camera.prim_path

    def annotator(self, name: str):
        return self.camera._custom_annotators[name]

    def get_data(self, name: str):
        return self.annotator(name).get_data()

    def has(self, name: str) -> bool:
        return name in self.camera._custom_annotators

    def ensure(self, name: str, **kwargs) -> None:
        if not self.has(name):
            self.camera.attach_annotator(name, **kwargs)

    def set_updates_enabled(self, enabled: bool) -> bool:
        rp = getattr(self.camera, "_render_product", None)
        tex = getattr(rp, "hydra_texture", None) if rp is not None else None
        if tex is None:
            return False
        tex.set_updates_enabled(bool(enabled))
        return True


class CameraRig:
    """Up to ``n_cams`` inspection cameras + an optional overview camera, created once.

    Typical use (worker)::

        rig.place([T_wc0, T_wc1, ...])   # activates cams 0..k-1, parks the others
        render_fresh(); views = capture_views(rig, ...)
    """

    def __init__(self, frame, sensor=None, n_cams: int = 6, overview: bool = True, root: str = "/World/Rig",
                 annotators=DEFAULT_ANNOTATORS, overview_resolution=(1280, 720), overview_hfov_deg: float = 50.0):
        from isaacsim.sensors.camera import Camera

        from ..schemas.sensor import SensorModel

        self.frame = frame
        self.sensor = sensor or SensorModel()
        self.pinhole = self.sensor.camera()
        self.root = root
        self.cams: list[RigCamera] = []
        for i in range(n_cams):
            cam = Camera(prim_path=f"{root}/Cam_{i}", name=f"isco_cam_{i}",
                         resolution=(self.sensor.width, self.sensor.height))
            cam.initialize(attach_rgb_annotator=False)
            set_pinhole_intrinsics(cam, self.pinhole, self.sensor.focal_length_mm, self.sensor.near_m,
                                   self.sensor.far_m)
            for name in annotators:
                if name == "semantic_segmentation":
                    cam.attach_annotator(name, colorize=False, semanticTypes=list(SEMANTIC_TYPES))
                else:
                    cam.attach_annotator(name)
            self.cams.append(RigCamera(i, cam, self.pinhole, annotators))
        self.overview: RigCamera | None = None
        if overview:
            w, h = overview_resolution
            ov_pin = PinholeCamera(int(w), int(h), float(overview_hfov_deg))
            cam = Camera(prim_path=f"{root}/Overview", name="isco_overview", resolution=(int(w), int(h)))
            cam.initialize(attach_rgb_annotator=True)
            set_pinhole_intrinsics(cam, ov_pin, 24.0, 0.05, 50.0)
            self.overview = RigCamera(-1, cam, ov_pin, ["rgb"])
            self.park(self.overview)
        for c in self.cams:
            self.park(c)

    # -- state -------------------------------------------------------------------------
    def configure(self, sensor) -> bool:
        """Switch all inspection cameras to ``sensor`` (resolution / FOV / focal length).
        Returns True if anything changed."""
        same = (sensor.width, sensor.height, sensor.hfov_deg, sensor.focal_length_mm, sensor.near_m, sensor.far_m) == (
            self.sensor.width, self.sensor.height, self.sensor.hfov_deg, self.sensor.focal_length_mm,
            self.sensor.near_m, self.sensor.far_m)
        if same:
            return False
        self.sensor = sensor
        self.pinhole = sensor.camera()
        for c in self.cams:
            if (sensor.width, sensor.height) != tuple(c.camera.get_resolution()):
                c.camera.set_resolution((sensor.width, sensor.height))
            set_pinhole_intrinsics(c.camera, self.pinhole, sensor.focal_length_mm, sensor.near_m, sensor.far_m)
            c.pinhole = self.pinhole
        return True

    @property
    def n_cams(self) -> int:
        return len(self.cams)

    @property
    def K(self) -> np.ndarray:
        return self.pinhole.K

    def park(self, cam: RigCamera) -> None:
        if not cam.active:
            return
        cam.set_updates_enabled(False)
        cam.camera.set_world_pose(position=PARK_POSITION_STAGE + np.array([0.0, 0.0, -cam.index]),
                                  orientation=np.array([1.0, 0.0, 0.0, 0.0]), camera_axes="usd")
        cam.active = False
        cam.T_world_cam = None

    def activate(self, cam: RigCamera) -> None:
        if cam.active:
            return
        cam.set_updates_enabled(True)
        cam.active = True

    def set_pose(self, i: int, T_world_cam: np.ndarray) -> RigCamera:
        cam = self.cams[i]
        self.activate(cam)
        set_camera_pose_cv(cam.camera, self.frame, T_world_cam)
        cam.T_world_cam = np.asarray(T_world_cam, dtype=float).copy()
        return cam

    def place(self, poses: list[np.ndarray]) -> list[RigCamera]:
        """Activate cameras 0..len(poses)-1 at ``poses`` (ISCO, OpenCV) and park the rest."""
        if len(poses) > self.n_cams:
            raise ValueError(f"{len(poses)} poses but the rig has {self.n_cams} cameras")
        out = [self.set_pose(i, T) for i, T in enumerate(poses)]
        for c in self.cams[len(poses):]:
            self.park(c)
        return out

    def park_all(self) -> None:
        for c in self.cams:
            self.park(c)
        if self.overview is not None:
            self.park(self.overview)

    def active_cams(self) -> list[RigCamera]:
        return [c for c in self.cams if c.active]

    def enable_rgb(self, cams: list[RigCamera] | None = None) -> None:
        for c in cams or self.cams:
            c.ensure("rgb")

    def place_overview(self, eye_world: np.ndarray, target_world: np.ndarray) -> RigCamera:
        if self.overview is None:
            raise RuntimeError("rig was built without an overview camera")
        T = look_at_opencv(np.asarray(eye_world, float), np.asarray(target_world, float))
        self.activate(self.overview)
        set_camera_pose_cv(self.overview.camera, self.frame, T)
        self.overview.T_world_cam = T
        return self.overview

    # -- checks ------------------------------------------------------------------------
    def isaac_K(self, i: int = 0) -> np.ndarray:
        return np.asarray(self.cams[i].camera.get_intrinsics_matrix(), dtype=float)

    def isaac_T_world_cam(self, i: int = 0) -> np.ndarray:
        """``T_world_cam`` (OpenCV axes, ISCO world) as Isaac sees it (from the view matrix)."""
        V = np.asarray(self.cams[i].camera.get_view_matrix_ros(), dtype=float)
        return self.frame.to_world_T(np.linalg.inv(V))


def add_conveyor_camera(center, size, resolution, distance_factor=1.6, focal_length=1.8, span=None,
                        prim_path="/World/Camera"):
    """PoC 3/4 overview camera framing the whole conveyor (or ``span`` metres around
    ``center``) - used by the demo video. ``focal_length`` is in Isaac stage units
    (``Camera.set_focal_length``; the default aperture makes 1.8 ~ 60 deg FOV)."""
    from isaacsim.sensors.camera import Camera

    from ..geometry.transforms import look_at_quat_wxyz_usd

    if span is None:
        span = float(np.max(size[:2]))
    target = np.asarray(center, dtype=float)
    eye = target + np.array([span * 0.9, -span * 1.05, span * 0.75]) * distance_factor
    camera = Camera(prim_path=prim_path, name="conveyor_camera", resolution=resolution)
    camera.initialize()
    camera.set_world_pose(position=eye, orientation=look_at_quat_wxyz_usd(eye, target), camera_axes="usd")
    camera.set_focal_length(focal_length)
    return camera


def add_inspection_cameras(center, num_cameras, part_size, radius_factor, elevation_deg, resolution,
                           focal_length=24.0, pointcloud=True, prim_root="/World/InspectionCameras"):
    """PoC camera ring: ``num_cameras`` cameras at equal azimuth steps, common elevation,
    radius ``part_size * radius_factor`` around ``center`` (stage frame). Note the PoC
    passes ``focal_length`` to ``Camera.set_focal_length`` (stage units: 24 -> 240 mm-
    equivalent with the default 20.955 aperture, i.e. ~5 deg FOV)."""
    from isaacsim.sensors.camera import Camera

    from ..geometry.transforms import look_at_quat_wxyz_usd

    radius = part_size * radius_factor
    elev = np.radians(elevation_deg)
    cameras = []
    for i in range(num_cameras):
        az = np.radians(360.0 * i / num_cameras)
        eye = np.asarray(center, float) + radius * np.array([np.cos(elev) * np.cos(az), np.cos(elev) * np.sin(az),
                                                             np.sin(elev)])
        camera = Camera(prim_path=f"{prim_root}/Cam_{i}", name=f"inspection_cam_{i}", resolution=resolution)
        camera.initialize()
        camera.set_world_pose(position=eye, orientation=look_at_quat_wxyz_usd(eye, center), camera_axes="usd")
        camera.set_focal_length(focal_length)
        if pointcloud:
            camera.add_pointcloud_to_frame()
        cameras.append(camera)
    return cameras
