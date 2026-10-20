"""CAD-distance baseline (ADR-0005): unsigned distance of each registered measured point
to the nominal CAD surface. Honest, training-free baseline the memory bank must beat
under realistic noise and normal part-to-part variation."""
from __future__ import annotations

import numpy as np

from ..geometry.mesh_io import LabeledMesh


class CadDistance:
    def __init__(self, template: LabeledMesh):
        import open3d as o3d

        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor(template.vertices.astype(np.float32)),
                                 o3d.core.Tensor(template.faces.astype(np.uint32)))
        self.o3d = o3d

    def point_scores(self, xyz: np.ndarray) -> np.ndarray:
        if len(xyz) == 0:
            return np.zeros(0, np.float32)
        return self.scene.compute_distance(self.o3d.core.Tensor(xyz.astype(np.float32))).numpy().astype(np.float32)
