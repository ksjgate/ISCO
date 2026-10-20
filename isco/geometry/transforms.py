"""Rigid-transform helpers and the ISCO frame conventions.

Frames (all units metres):

* **world** - ISCO's process frame. +Z up, +X = belt travel direction, the belt top
  surface is the plane z = 0 and the nominal part placement (focus point) sits above
  the origin. (Isaac Sim's stage frame differs only by a translation; see
  ``isco.sim.scene``.)
* **part / rest frame** - the frame a part mesh is stored in: the part rests in its
  nominal stable pose with its lowest point at z = 0 and its bounding-box centre at
  x = y = 0. The nominal pose is therefore ``T_world_part = I``.
* **camera (OpenCV)** - x right, y down, z forward (optical axis). ``T_world_cam``
  maps camera coordinates to world coordinates; its rotation columns are the camera
  axes expressed in world coordinates.
* **camera (USD)** - x right, y up, looks along -z. Only ``isco.sim`` needs it.

Quaternions are scalar-first ``(w, x, y, z)`` like Isaac Sim's APIs.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

WORLD_UP = np.array([0.0, 0.0, 1.0])


def make_T(R: np.ndarray | None = None, t: np.ndarray | None = None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = np.asarray(t, dtype=float)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def apply_T(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to (N,3) points."""
    pts = np.asarray(pts)
    return pts @ T[:3, :3].T + T[:3, 3]


def apply_R(T_or_R: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    R = T_or_R[:3, :3]
    return np.asarray(vecs) @ R.T


def rot_x(deg: float) -> np.ndarray:
    return Rotation.from_euler("x", deg, degrees=True).as_matrix()


def rot_y(deg: float) -> np.ndarray:
    return Rotation.from_euler("y", deg, degrees=True).as_matrix()


def rot_z(deg: float) -> np.ndarray:
    return Rotation.from_euler("z", deg, degrees=True).as_matrix()


def look_at_opencv(eye: np.ndarray, target: np.ndarray, up: np.ndarray = WORLD_UP) -> np.ndarray:
    """``T_world_cam`` (OpenCV axes) of a camera at ``eye`` looking at ``target``.

    Columns of the rotation are (right, down, forward). When the viewing direction
    is (anti)parallel to ``up`` the world +X axis is used to fix the roll so that
    image "up" points towards -X (belt upstream).
    """
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    f = target - eye
    n = np.linalg.norm(f)
    if n < 1e-12:
        raise ValueError("eye and target coincide")
    f /= n
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-8:  # looking straight up/down
        r = np.cross(f, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(r) < 1e-8:
            r = np.array([0.0, 1.0, 0.0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f], axis=1)
    return make_T(R, eye)


def opencv_to_usd_rotation(R_cv: np.ndarray) -> np.ndarray:
    """Camera rotation with OpenCV axes -> same camera with USD axes (y up, -z fwd)."""
    return R_cv @ np.diag([1.0, -1.0, -1.0])


def quat_wxyz_from_R(R: np.ndarray) -> np.ndarray:
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def R_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def look_at_quat_wxyz_usd(eye: np.ndarray, target: np.ndarray, up: np.ndarray = WORLD_UP) -> np.ndarray:
    """Gf-free replacement of the PoC ``look_at_quat_wxyz`` (USD camera axes)."""
    T = look_at_opencv(eye, target, up)
    return quat_wxyz_from_R(opencv_to_usd_rotation(T[:3, :3]))


def pose_from_params(dx=0.0, dy=0.0, dz=0.0, yaw_deg=0.0, roll_deg=0.0, pitch_deg=0.0) -> np.ndarray:
    """Part pose relative to its nominal rest pose (rotation about the rest-frame origin)."""
    R = rot_z(yaw_deg) @ rot_y(pitch_deg) @ rot_x(roll_deg)
    return make_T(R, [dx, dy, dz])


def rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(R).as_rotvec())))
