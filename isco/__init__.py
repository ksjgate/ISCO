"""ISCO - Inspection Setup Configuration Optimizer.

Detector-in-the-loop search of inline 3D inspection cell configurations
(camera count/pose, rotation stage, belt speed, trigger scheme) in an Isaac Sim
conveyor digital twin.

Import rule: only ``isco.sim`` may import ``isaacsim`` / ``pxr`` / ``omni`` (and only
lazily, inside functions, after ``SimulationApp`` exists). Everything else must run
without Isaac Sim loaded.
"""

__version__ = "0.1.0"
