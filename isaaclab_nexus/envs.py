"""Run Isaac Lab manager-based envs on the Nexus backend.

`nexusify(env_cfg, mjcf_path)` rewrites an existing `ManagerBasedRLEnvCfg` in place:
physics -> `NexusCfg`; every `ArticulationCfg` spawn -> `NexusMjcfCfg` (MJCF replaces the USD);
visual-only scene assets dropped; event terms that need PhysX-only APIs removed (listed).
`install()` makes `ManagerBasedEnv` build a `NexusScene` instead of `InteractiveScene`
whenever the sim cfg selects Nexus.
"""

from __future__ import annotations

import warnings

from isaaclab.assets import ArticulationCfg, AssetBaseCfg

from .physics.nexus_manager_cfg import NexusCfg, NexusMjcfCfg

# event / curriculum term functions with no Nexus equivalent (need PhysX views); dropped with a warning
_UNSUPPORTED_TERMS = {"randomize_rigid_body_com", "randomize_rigid_body_material"}
_installed = False


def install() -> None:
    """Route `ManagerBasedEnv`'s scene construction to `NexusScene` when the physics cfg is Nexus."""
    global _installed
    if _installed:
        return
    import isaaclab.envs.manager_based_env as mbe
    from isaaclab.scene import InteractiveScene
    from .scene import NexusScene

    class _SceneDispatch:
        def __new__(cls, cfg):
            from isaaclab.sim import SimulationContext
            sim = SimulationContext.instance()
            if sim is not None and isinstance(getattr(sim.cfg, "physics", None), NexusCfg):
                return NexusScene(cfg)
            return InteractiveScene(cfg)

    mbe.InteractiveScene = _SceneDispatch
    _installed = True


def nexusify(env_cfg, mjcf_path: str, *, collisions_capacity: int = 256, solver_iterations: int = 4, drop_terms: set[str] | None = None):
    install()
    env_cfg.sim.physics = NexusCfg(collisions_capacity=collisions_capacity, solver_iterations=solver_iterations)
    scene = env_cfg.scene
    for name in list(vars(scene)):
        if name.startswith("_"):
            continue
        ecfg = getattr(scene, name)
        if isinstance(ecfg, ArticulationCfg):
            ecfg.spawn = NexusMjcfCfg(mjcf_path=mjcf_path, num_envs=scene.num_envs, auto_floor=False)
        elif isinstance(ecfg, AssetBaseCfg) and not isinstance(ecfg, ArticulationCfg):
            setattr(scene, name, None)                                   # lights etc.
    dropped = []
    for group in ("events", "curriculum", "rewards", "terminations"):
        g = getattr(env_cfg, group, None)
        if g is None:
            continue
        for name in list(vars(g)):
            term = getattr(g, name)
            fn = getattr(term, "func", None)
            fname = getattr(fn, "__name__", type(fn).__name__ if fn is not None else "")
            if fname in _UNSUPPORTED_TERMS or (drop_terms and name in drop_terms):
                setattr(g, name, None); dropped.append(f"{group}.{name} ({fname})")
    if dropped:
        warnings.warn("nexusify dropped terms without a Nexus implementation: " + ", ".join(dropped))
    env_cfg.sim.render_interval = env_cfg.decimation
    return env_cfg
