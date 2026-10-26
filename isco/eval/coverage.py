"""PoC reconstruction-coverage metrics (moved out of the original
``create_conveyor.py``; pure functions, no Isaac Sim needed - plan "PoC 리팩터 매핑").

* ``compare_reconstruction``: completeness (GT surface -> reconstruction) and accuracy
  (reconstruction -> GT surface) at a distance threshold, Chamfer-style.
* ``visualize_coverage``: per-camera points + completeness heat map figure.
"""
from __future__ import annotations

import numpy as np


def point_mesh_distance(points: np.ndarray, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Exact unsigned point-to-triangle-mesh distance (open3d if importable, else trimesh)."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return np.zeros(0)
    try:
        import open3d as o3d

        # open3d works in float32: shift to the mesh centre to keep precision for stage coords
        c = np.asarray(vertices, np.float64).mean(axis=0)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor((np.asarray(vertices, np.float64) - c).astype(np.float32)),
                            o3d.core.Tensor(np.asarray(faces, np.uint32)))
        return scene.compute_distance(o3d.core.Tensor((points - c).astype(np.float32))).numpy().astype(np.float64)
    except Exception:
        import trimesh

        tm = trimesh.Trimesh(vertices, faces, process=False)
        _, d, _ = trimesh.proximity.closest_point(tm, points)
        return np.asarray(d, dtype=np.float64)


def compare_reconstruction(gt_verts_world, gt_faces, merged_points, threshold, num_samples=8000, seed=0,
                           fine_threshold=0.2e-3):
    """PoC completeness/accuracy (Chamfer-style, both directions) + exact fine metrics.

    * ``completeness`` / ``accuracy`` (PoC definition): fraction of GT surface samples
      within ``threshold`` of a reconstructed point / of reconstructed points within
      ``threshold`` of a GT surface *sample* (area-weighted samples, not vertices).
    * ``accuracy_exact@thr``: reconstructed points within ``thr`` of the GT *mesh*
      (exact point-to-triangle distance) - needed at 0.2 mm where the sample spacing
      (~0.6 mm for 8000 samples) would dominate.
    * ``completeness@0.2mm`` uses the same GT samples with the fine threshold.
    """
    import trimesh
    from scipy.spatial import cKDTree

    mesh = trimesh.Trimesh(vertices=gt_verts_world, faces=gt_faces, process=False)
    gt_surface_points, _ = trimesh.sample.sample_surface(mesh, num_samples, seed=seed)
    merged_points = np.asarray(merged_points, dtype=float).reshape(-1, 3)
    result = {
        "gt_surface_points": gt_surface_points,
        "completeness": 0.0,
        "completeness_dists": np.full(len(gt_surface_points), np.inf),
        "accuracy": 0.0,
        "accuracy_dists": np.full(len(merged_points), np.inf),
        "mean_completeness_dist": float("inf"),
        "mean_accuracy_dist": float("inf"),
        "threshold": float(threshold),
        "fine_threshold": float(fine_threshold),
    }
    if len(merged_points) == 0:
        result.update({"completeness_fine": 0.0, "accuracy_exact": 0.0, "accuracy_exact_fine": 0.0,
                       "accuracy_exact_dists": np.zeros(0)})
        return result
    comp_d, _ = cKDTree(merged_points).query(gt_surface_points)
    acc_d, _ = cKDTree(gt_surface_points).query(merged_points)
    exact = point_mesh_distance(merged_points, gt_verts_world, gt_faces)
    result.update(
        completeness_dists=comp_d, completeness=float(np.mean(comp_d < threshold)),
        mean_completeness_dist=float(np.mean(comp_d)),
        accuracy_dists=acc_d, accuracy=float(np.mean(acc_d < threshold)), mean_accuracy_dist=float(np.mean(acc_d)),
        completeness_fine=float(np.mean(comp_d < fine_threshold)),
        accuracy_exact=float(np.mean(exact < threshold)), accuracy_exact_fine=float(np.mean(exact < fine_threshold)),
        accuracy_exact_dists=exact, accuracy_exact_median_m=float(np.median(exact)),
        accuracy_exact_p99_m=float(np.percentile(exact, 99)),
    )
    return result


def visualize_coverage(gt_verts_world, gt_faces, per_camera_points, comparison, threshold, out_path):
    """PoC 2x2 figure: per-camera points over the GT mesh + completeness heat map, from
    two viewing angles (green = seen, red = blind spot)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    for _font in ["Noto Sans CJK KR", "NanumGothic", "Malgun Gothic"]:
        if _font in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
            matplotlib.rcParams["font.family"] = _font
            break
    gt_verts_world = np.asarray(gt_verts_world)
    mins, maxs = gt_verts_world.min(axis=0), gt_verts_world.max(axis=0)
    pad = float((maxs - mins).max()) * 0.6 + 1e-4
    center = (mins + maxs) / 2.0
    cam_colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"]
    views = [(20, -60), (75, -90)]
    fig = plt.figure(figsize=(15, 5.5 * len(views)))
    for row, (elev, azim) in enumerate(views):
        ax1 = fig.add_subplot(len(views), 2, row * 2 + 1, projection="3d")
        ax1.add_collection3d(Poly3DCollection(gt_verts_world[gt_faces], alpha=0.15, facecolor="gray",
                                              edgecolor="gray", linewidths=0.1))
        for i, pts in enumerate(per_camera_points):
            if len(pts) == 0:
                continue
            sub = pts[:: max(1, len(pts) // 20000)]
            ax1.scatter(sub[:, 0], sub[:, 1], sub[:, 2], s=2, alpha=0.5, color=cam_colors[i % len(cam_colors)],
                        label=f"Cam {i}")
        if row == 0:
            ax1.legend(loc="upper left", fontsize=8, markerscale=3)
            ax1.set_title("per-camera reconstructed points", fontsize=10)
        ax2 = fig.add_subplot(len(views), 2, row * 2 + 2, projection="3d")
        gt_pts = comparison["gt_surface_points"]
        dists = np.clip(comparison["completeness_dists"], 0, threshold * 4)
        sc = ax2.scatter(gt_pts[:, 0], gt_pts[:, 1], gt_pts[:, 2], c=dists, cmap="RdYlGn_r", s=4, vmin=0,
                         vmax=threshold * 4)
        if row == 0:
            ax2.set_title("GT surface -> nearest reconstructed point (green = seen, red = blind spot)", fontsize=10)
            cbar = fig.colorbar(sc, ax=ax2, shrink=0.6, pad=0.02)
            cbar.set_label("distance (m)", fontsize=8)
        for ax in (ax1, ax2):
            ax.set_xlim(center[0] - pad, center[0] + pad)
            ax.set_ylim(center[1] - pad, center[1] + pad)
            ax.set_zlim(center[2] - pad, center[2] + pad)
            ax.view_init(elev=elev, azim=azim)
            ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------- PoC capture
