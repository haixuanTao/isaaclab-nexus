"""Ablation + NVTX instrumentation for WBC-AGILE, applied by monkey-patch.

Nothing in the WBC-AGILE clone or the venv is modified; this module is imported
by train_ablate.py after Isaac Lab is loaded but before the env is constructed.

Env vars
--------
AGILE_ABLATE : none (default) | nophysics | nophysics_noreadback
    nophysics            -> SimulationContext.step() becomes a no-op.
                            Removes the PhysX solve; keeps action processing,
                            buffer writes, sensor/state readback, MDP terms,
                            policy inference and the PPO update.
    nophysics_noreadback -> additionally InteractiveScene.update() is a no-op,
                            removing the per-step sensor/view readback.

AGILE_NVTX : 0 (default) | 1
    Wrap the pipeline's phase boundaries in NVTX ranges so nsys can cut the
    trace on real markers instead of inferred kernel patterns.

NOTE ON VALIDITY: with physics stubbed the robots never move, so contacts stay
zero and terminations stop firing. This measures the COST of the surrounding
pipeline on frozen state, not a physically meaningful rollout. Branch-dependent
work (resets, curriculum) will differ from the live run; see FINDINGS.
"""

import os

_ABLATE = os.environ.get("AGILE_ABLATE", "none").lower()
_NVTX = os.environ.get("AGILE_NVTX", "0") == "1"

_applied = []


def _wrap_nvtx(cls, meth, label):
    """Wrap cls.meth in an NVTX range named `label`."""
    import torch

    orig = getattr(cls, meth, None)
    if orig is None:
        return False

    def wrapped(*a, **kw):
        torch.cuda.nvtx.range_push(label)
        try:
            return orig(*a, **kw)
        finally:
            torch.cuda.nvtx.range_pop()

    wrapped.__name__ = getattr(orig, "__name__", meth)
    setattr(cls, meth, wrapped)
    return True


def apply():
    if _ABLATE == "none" and not _NVTX:
        print("[ablate] no-op (AGILE_ABLATE=none, AGILE_NVTX=0)")
        return

    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationContext

    # ---------------- ablation ----------------
    if _ABLATE.startswith("nophysics"):
        def _noop_step(self, render=False):
            return None

        SimulationContext.step = _noop_step
        _applied.append("SimulationContext.step -> no-op (PhysX solve removed)")

        if "noreadback" in _ABLATE:
            def _noop_update(self, dt):
                return None

            InteractiveScene.update = _noop_update
            _applied.append("InteractiveScene.update -> no-op (sensor readback removed)")
    elif _ABLATE != "none":
        raise ValueError(f"unknown AGILE_ABLATE={_ABLATE!r}")

    # ---------------- nvtx ----------------
    if _NVTX:
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab.managers import (
            ActionManager,
            CommandManager,
            ObservationManager,
            RewardManager,
            TerminationManager,
        )

        targets = [
            (ActionManager, "process_action", "act/process"),
            (ActionManager, "apply_action", "act/apply"),
            (InteractiveScene, "write_data_to_sim", "sim/write"),
            (SimulationContext, "step", "sim/step"),
            (InteractiveScene, "update", "sim/readback"),
            (CommandManager, "compute", "mdp/command"),
            (RewardManager, "compute", "mdp/reward"),
            (TerminationManager, "compute", "mdp/termination"),
            (ObservationManager, "compute", "mdp/observation"),
            (ManagerBasedRLEnv, "_reset_idx", "mdp/reset"),
            (ManagerBasedRLEnv, "step", "env/step"),
        ]
        for cls, meth, label in targets:
            if _wrap_nvtx(cls, meth, label):
                _applied.append(f"nvtx {label} <- {cls.__name__}.{meth}")

    print("[ablate] " + f"AGILE_ABLATE={_ABLATE} AGILE_NVTX={int(_NVTX)}")
    for line in _applied:
        print(f"[ablate]   {line}")


apply()
