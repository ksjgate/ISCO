"""Embree (open3d RaycastingScene) depth renderer - the bulk renderer of the slot cache
(ADR-0003, D2). It reproduces Isaac Sim's ``distance_to_image_plane`` geometry exactly
(noise-free ray depth, same pinhole/pixel-centre conventions); equivalence is checked by
experiment E0.1 against the Isaac worker.
"""
from __future__ import annotations

import numpy as np

from ..geometry.camera_model import PinholeCamera
from ..geometry.labels import pixel_label_from_face_label
from ..geometry.mesh_io import LabeledMesh
from ..geometry.transforms import apply_T
from .occluder import Occluder
from .pixels import ViewPixels

_MAX_RAYS_PER_CALL = 8_000_000


class RaycastRenderer:
    def __init__(self, camera: PinholeCamera, occluder: Occluder | None = None, n_threads: int = 0):
        self.camera = camera
        self.occluder = occluder
        self.n_threads = n_threads

    def _scene(self, mesh: LabeledMesh, T_world_part: np.ndarray):
        import open3d as o3d

        scene = o3d.t.geometry.RaycastingScene(nthreads=self.n_threads)
        v = apply_T(np.asarray(T_world_part), mesh.vertices).astype(np.float32)
        part_gid = scene.add_triangles(o3d.core.Tensor(v), o3d.core.Tensor(mesh.faces.astype(np.uint32)))
        if self.occluder is not None:
            scene.add_triangles(o3d.core.Tensor(self.occluder.vertices.astype(np.float32)),
                                o3d.core.Tensor(self.occluder.faces.astype(np.uint32)))
        return scene, part_gid

    def render(self, mesh: LabeledMesh, T_world_part: np.ndarray,
               views: list[tuple[np.ndarray, tuple[int, int, int, int]]], metas: list[dict] | None = None) -> list[ViewPixels]:
        """Render ``mesh`` placed at ``T_world_part`` from each ``(T_world_cam, roi)``."""
        import open3d as o3d

        cam = self.camera
        scene, part_gid = self._scene(mesh, T_world_part)
        face_pix_label = pixel_label_from_face_label(mesh.face_label)
        # assemble all rays, cast in chunks
        per_view = []
        for (T_wc, roi) in views:
            pix = cam.roi_pixels(roi)
            d_cam = cam.ray_dirs_cam(pix)
            norm = np.linalg.norm(d_cam, axis=1)
            d_cam_n = d_cam / norm[:, None]
            d_w = d_cam_n @ T_wc[:3, :3].T
            o_w = np.broadcast_to(T_wc[:3, 3], d_w.shape)
            per_view.append((pix, norm, d_w, o_w))
        rays = np.concatenate([np.concatenate([o, d], axis=1) for (_, _, d, o) in per_view], axis=0).astype(np.float32)
        t_hit = np.empty(len(rays), np.float32)
        gid = np.empty(len(rays), np.int64)
        prim = np.empty(len(rays), np.int64)
        nrm = np.empty((len(rays), 3), np.float32)
        for s in range(0, len(rays), _MAX_RAYS_PER_CALL):
            e = min(s + _MAX_RAYS_PER_CALL, len(rays))
            ans = scene.cast_rays(o3d.core.Tensor(rays[s:e]), nthreads=self.n_threads)
            t_hit[s:e] = ans["t_hit"].numpy()
            gid[s:e] = ans["geometry_ids"].numpy().astype(np.int64)
            prim[s:e] = ans["primitive_ids"].numpy().astype(np.int64)
            nrm[s:e] = ans["primitive_normals"].numpy()
        out = []
        off = 0
        for i, (pix, norm, d_w, _) in enumerate(per_view):
            n = len(pix)
            sl = slice(off, off + n)
            off += n
            hit = (gid[sl] == part_gid) & np.isfinite(t_hit[sl])
            depth = (t_hit[sl][hit] / norm[hit]).astype(np.float32)   # ray length -> z-depth
            cosv = np.abs(np.einsum("ij,ij->i", nrm[sl][hit], d_w[hit])).astype(np.float32)
            label = face_pix_label[prim[sl][hit]]
            meta = dict(metas[i]) if metas else {}
            out.append(ViewPixels(pix[hit].astype(np.int64), depth, label, cosv, meta))
        return out

    def render_full_depth(self, mesh: LabeledMesh, T_world_part: np.ndarray, T_world_cam: np.ndarray,
                          roi: tuple[int, int, int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Dense depth (0 = no hit on anything) and label image over ``roi`` - debugging/figures."""
        import open3d as o3d

        cam = self.camera
        roi = roi or (0, 0, cam.width, cam.height)
        pix = cam.roi_pixels(roi)
        d_cam = cam.ray_dirs_cam(pix)
        norm = np.linalg.norm(d_cam, axis=1)
        d_w = (d_cam / norm[:, None]) @ T_world_cam[:3, :3].T
        rays = np.concatenate([np.broadcast_to(T_world_cam[:3, 3], d_w.shape), d_w], axis=1).astype(np.float32)
        scene, part_gid = self._scene(mesh, T_world_part)
        ans = scene.cast_rays(o3d.core.Tensor(rays))
        t = ans["t_hit"].numpy()
        g = ans["geometry_ids"].numpy().astype(np.int64)
        p = ans["primitive_ids"].numpy().astype(np.int64)
        depth = np.where(np.isfinite(t), t / norm, 0.0).reshape(roi[3] - roi[1], roi[2] - roi[0])
        lab = np.zeros(len(t), np.uint8)
        hit = (g == part_gid) & np.isfinite(t)
        lab[hit] = pixel_label_from_face_label(mesh.face_label)[p[hit]]
        return depth.astype(np.float32), lab.reshape(depth.shape)
