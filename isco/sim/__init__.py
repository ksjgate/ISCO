"""Isaac Sim side of ISCO (the only package that imports ``isaacsim`` / ``omni`` / ``pxr``).

All Isaac imports are lazy (inside functions), so ``import isco.sim`` and its submodules
work without a running ``SimulationApp``. Launch the app first with
``isco.sim.app.launch()`` before calling anything that touches the stage.

Modules: ``app`` (SimulationApp lifecycle), ``scene`` (conveyor scene, ISCO<->stage
frame), ``part_prim`` (labelled part meshes, rigid bodies), ``rig`` (camera rig),
``capture`` (fresh rendering, ViewPixels, verify_pose, PoC capture), ``process``
(physics settling), ``export_occluder`` (conveyor geometry for embree), ``worker``
(job-queue render worker, ``python -m isco.sim.worker``).
"""
