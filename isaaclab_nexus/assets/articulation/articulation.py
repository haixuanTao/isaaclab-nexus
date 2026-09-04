"""Nexus-backed ``Articulation`` and ``ArticulationData`` (fork, CUDA).

State is served as **zero-copy torch views** over the simulator's GPU memory
(``__cuda_array_interface__``), wrapped in Isaac Lab's ``ProxyArray`` so both
``data.joint_pos.torch`` and plain tensor access work. The per-link SoA
workspace has index ``(link * WS_QUADS + quad) * num_batches + batch`` (batch
innermost), so wrapped as ``(links, WS_QUADS, num_batches, 4)`` every read
below is a strided view; the only per-call kernels are the flat-DOF gathers.

Conventions (validated in bench/nexus_port/test_write_path.py, test_g1_articulation.py):
  * the floating base is a 6-DOF free joint on link 0 and is EXCLUDED from
    ``joint_*`` vectors, as in Isaac Lab; ``num_joints`` counts articulated DOFs;
  * joint order = ascending generalized-DOF column (``assembly_id``);
  * per-link coordinates: 6 floats, quad WS_COORDS cols 0-3 + quad WS_COORDS+1 cols 0-1;
    a 1-DOF joint's coordinate index == its free axis index (3..5 for hinges);
    the free root: coords 0-2 = world position, orientation = quad WS_JOINT_ROT (xyzw);
  * generalized velocities: ``dof_state[0]`` (D, NB); the root's 6 = linear (world) then
    angular (world), from the integrator's ``disp * joint_rot`` update;
  * quaternions are (x, y, z, w) in both Nexus and Isaac Lab 3.0 -- passed through unchanged;
  * ACTUATION: Isaac Lab's actuator models (Implicit / DCMotor / Delayed...) run in Python
    exactly as on PhysX; the resulting ``applied_effort`` is written as a generalized force
    (``external_gen_forces``) every ``write_data_to_sim``. Engine motors are disabled (gains 0)
    so implicit actuators become explicit PD at the physics rate;
  * body wrenches (``permanent_wrench_composer``) are projected onto the root free joint:
    F_root = sum F_b, tau_root = sum (tau_b + (p_b - p_root) x F_b). This is the base-DOF part of
    J^T w; the joint-space part is dropped (stated approximation);
  * resets restore the published template (the placed initial state) per env.
"""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Sequence
from typing import Any

import torch
import os
import warp as wp

from isaaclab.assets.articulation.base_articulation import BaseArticulation
from isaaclab.utils.warp.proxy_array import ProxyArray
from isaaclab.utils.types import ArticulationActions
from isaaclab.utils import math as math_utils

from ...physics.nexus_manager import NexusManager

_DEV = "cuda:0"


def _proxy(t: torch.Tensor) -> ProxyArray:
    """Wrap a CUDA torch tensor as an Isaac Lab ProxyArray without copying."""
    return ProxyArray(wp.from_torch(t if t.is_contiguous() else t.contiguous()))


def _match(names: list[str], keys, preserve_order: bool = False) -> tuple[list[int], list[str]]:
    keys = [keys] if isinstance(keys, str) else list(keys)
    pats = [re.compile(k) for k in keys]
    if preserve_order:
        idx = [i for p in pats for i, n in enumerate(names) if p.fullmatch(n)]
    else:
        idx = [i for i, n in enumerate(names) if any(p.fullmatch(n) for p in pats)]
    seen, out = set(), []
    for i in idx:
        if i not in seen:
            seen.add(i); out.append(i)
    return out, [names[i] for i in out]


def _xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Identity. The engine stores quaternions as (x, y, z, w) and so does Isaac Lab 3.0 (see
    `isaaclab.utils.math.quat_apply_inverse`: "quaternion in (x, y, z, w)", `InitialStateCfg.rot`
    default (0, 0, 0, 1), `isaaclab_physx` ArticulationData). This backend was first written to the
    pre-3.0 (w, x, y, z) convention; the conversion scrambled every quaternion crossing the boundary
    -- robots spawned yawed 180 degrees and every body-frame observation of a non-upright robot was
    computed from the wrong rotation. Kept as a named no-op so every call site stays visible."""
    return q

def _wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    """Identity -- see `_xyzw_to_wxyz`."""
    return q

def _t(x) -> torch.Tensor:
    if isinstance(x, ProxyArray):
        return x.torch
    if isinstance(x, wp.array):
        return wp.to_torch(x)
    return torch.as_tensor(x, device=_DEV)


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (..., 3) by unit quaternions q (..., 4) in (x, y, z, w) order (Isaac Lab 3.0 / engine)."""
    xyz, w = q[..., :3], q[..., 3:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


class ArticulationData:
    """Isaac Lab ``ArticulationData`` surface over Nexus state.

    Views straight into the simulator where a field is a strided view of the workspace;
    small derived tensors (body-frame quantities, quaternion reorders) are computed on access.
    """

    GRAVITY_VEC_W = None
    FORWARD_VEC_B = None

    def __init__(self, art: "Articulation"):
        a = self._a = art
        N, B, J = art.num_instances, art.num_bodies, art.num_joints
        self.num_instances, self.num_bodies, self.num_joints = N, B, J
        self.body_names, self.joint_names = art.body_names, art.joint_names
        self.fixed_tendon_names = []
        self._grav_w = torch.tensor([0.0, 0.0, -1.0], device=_DEV).expand(N, 3).contiguous()
        self._fwd_b = torch.tensor([1.0, 0.0, 0.0], device=_DEV).expand(N, 3).contiguous()
        self.GRAVITY_VEC_W = _proxy(self._grav_w); self.FORWARD_VEC_B = _proxy(self._fwd_b)
        z = lambda *s: torch.zeros(*s, device=_DEV)
        # defaults (filled by Articulation after placement)
        self._default_joint_pos = z(N, J); self._default_joint_vel = z(N, J)
        self._default_root_pose = z(N, 7); self._default_root_pose[:, 3] = 1.0
        self._default_root_vel = z(N, 6)
        # targets / actuator outputs
        self._joint_pos_target = z(N, J); self._joint_vel_target = z(N, J); self._joint_effort_target = z(N, J)
        self._computed_torque = z(N, J); self._applied_torque = z(N, J)
        self._joint_acc = z(N, J)
        # joint properties (sim-side "written" values; the engine uses limits from the MJCF)
        self._joint_stiffness = z(N, J); self._joint_damping = z(N, J); self._joint_armature = z(N, J)
        self._joint_friction_coeff = z(N, J); self._joint_dynamic_friction_coeff = z(N, J); self._joint_viscous_friction_coeff = z(N, J)
        self._joint_effort_limits = torch.full((N, J), float("inf"), device=_DEV)
        self._joint_vel_limits = torch.full((N, J), float("inf"), device=_DEV)
        self._joint_pos_limits = z(N, J, 2); self._joint_pos_limits[..., 0] = -math.pi; self._joint_pos_limits[..., 1] = math.pi
        self._soft_joint_pos_limit_factor = float(getattr(art.cfg, "soft_joint_pos_limit_factor", 1.0))
        # body properties
        self._body_mass = z(N, B); self._body_inertia = z(N, B, 9); self._body_com_pose_b = z(N, B, 7); self._body_com_pose_b[..., 3] = 1.0

    # ---- joint state ----
    @property
    def joint_pos(self): return _proxy(self._a._q_flat())
    @property
    def joint_vel(self): return _proxy(self._a._v_flat())
    @property
    def joint_acc(self): return _proxy(self._joint_acc)
    @property
    def joint_pos_target(self): return _proxy(self._joint_pos_target)
    @property
    def joint_vel_target(self): return _proxy(self._joint_vel_target)
    @property
    def joint_effort_target(self): return _proxy(self._joint_effort_target)
    @property
    def computed_torque(self): return _proxy(self._computed_torque)
    @property
    def applied_torque(self): return _proxy(self._applied_torque)
    @property
    def default_joint_pos(self): return _proxy(self._default_joint_pos)
    @property
    def default_joint_vel(self): return _proxy(self._default_joint_vel)
    @property
    def default_joint_pos_nominal(self): return _proxy(self._default_joint_pos)
    @property
    def joint_stiffness(self): return _proxy(self._joint_stiffness)
    @property
    def joint_damping(self): return _proxy(self._joint_damping)
    @property
    def joint_armature(self): return _proxy(self._joint_armature)
    @property
    def default_joint_armature(self): return _proxy(self._joint_armature)
    @property
    def joint_friction_coeff(self): return _proxy(self._joint_friction_coeff)
    @property
    def default_joint_friction_coeff(self): return _proxy(self._joint_friction_coeff)
    @property
    def joint_dynamic_friction_coeff(self): return _proxy(self._joint_dynamic_friction_coeff)
    @property
    def joint_viscous_friction_coeff(self): return _proxy(self._joint_viscous_friction_coeff)
    @property
    def joint_effort_limits(self): return _proxy(self._joint_effort_limits)
    @property
    def joint_vel_limits(self): return _proxy(self._joint_vel_limits)
    @property
    def joint_velocity_limits(self): return _proxy(self._joint_vel_limits)
    @property
    def joint_pos_limits(self): return _proxy(self._joint_pos_limits)
    @property
    def default_joint_pos_limits(self): return _proxy(self._joint_pos_limits)
    @property
    def soft_joint_pos_limits(self):
        lo, hi = self._joint_pos_limits[..., 0], self._joint_pos_limits[..., 1]
        m, r = 0.5 * (lo + hi), (hi - lo) * self._soft_joint_pos_limit_factor
        return _proxy(torch.stack([m - 0.5 * r, m + 0.5 * r], -1))
    @property
    def soft_joint_vel_limits(self): return _proxy(self._joint_vel_limits)

    # ---- root state (link 0), Isaac Lab 3.0 conventions: pos xyz, quat (x, y, z, w), vel [lin, ang] world ----
    @property
    def root_link_pos_w(self): return _proxy(self._a._quad(self._a._lay["WS_LTW"] + 1)[:, 0, :3])
    @property
    def root_link_quat_w(self): return _proxy(_xyzw_to_wxyz(self._a._quad(self._a._lay["WS_LTW"])[:, 0, :]))
    @property
    def root_link_pose_w(self): return _proxy(torch.cat([self.root_link_pos_w.torch, self.root_link_quat_w.torch], -1))
    @property
    def root_link_vel_w(self): return _proxy(self._a._vel(self._a._lay["WS_RB_VELS"])[:, 0, :])
    @property
    def root_link_lin_vel_w(self): return _proxy(self._a._quad(self._a._lay["WS_RB_VELS"])[:, 0, :3])
    @property
    def root_link_ang_vel_w(self): return _proxy(self._a._quad(self._a._lay["WS_RB_VELS"] + 1)[:, 0, :3])
    root_pos_w = root_link_pos_w; root_quat_w = root_link_quat_w; root_pose_w = root_link_pose_w
    root_vel_w = root_link_vel_w; root_lin_vel_w = root_link_lin_vel_w; root_ang_vel_w = root_link_ang_vel_w
    root_com_pos_w = root_link_pos_w; root_com_quat_w = root_link_quat_w; root_com_pose_w = root_link_pose_w
    root_com_vel_w = root_link_vel_w; root_com_lin_vel_w = root_link_lin_vel_w; root_com_ang_vel_w = root_link_ang_vel_w
    @property
    def root_state_w(self): return _proxy(torch.cat([self.root_link_pose_w.torch, self.root_link_vel_w.torch], -1))
    root_link_state_w = root_state_w; root_com_state_w = root_state_w
    @property
    def root_lin_vel_b(self): return _proxy(math_utils.quat_apply_inverse(self.root_quat_w.torch, self.root_lin_vel_w.torch))
    @property
    def root_ang_vel_b(self): return _proxy(math_utils.quat_apply_inverse(self.root_quat_w.torch, self.root_ang_vel_w.torch))
    root_link_lin_vel_b = root_lin_vel_b; root_link_ang_vel_b = root_ang_vel_b; root_com_lin_vel_b = root_lin_vel_b; root_com_ang_vel_b = root_ang_vel_b
    @property
    def projected_gravity_b(self): return _proxy(math_utils.quat_apply_inverse(self.root_quat_w.torch, self._grav_w))
    @property
    def heading_w(self):
        f = math_utils.quat_apply(self.root_quat_w.torch, self._fwd_b)
        return _proxy(torch.atan2(f[:, 1], f[:, 0]))
    @property
    def default_root_pose(self): return _proxy(self._default_root_pose)
    @property
    def default_root_vel(self): return _proxy(self._default_root_vel)
    @property
    def default_root_state(self): return _proxy(torch.cat([self._default_root_pose, self._default_root_vel], -1))

    # ---- body state ----
    @property
    def body_link_pos_w(self): return _proxy(self._a._quad(self._a._lay["WS_LTW"] + 1)[..., :3])
    @property
    def body_link_quat_w(self): return _proxy(_xyzw_to_wxyz(self._a._quad(self._a._lay["WS_LTW"])))
    @property
    def body_link_pose_w(self): return _proxy(torch.cat([self.body_link_pos_w.torch, self.body_link_quat_w.torch], -1))
    @property
    def body_link_vel_w(self): return _proxy(self._a._vel(self._a._lay["WS_RB_VELS"]))
    @property
    def body_link_lin_vel_w(self): return _proxy(self._a._quad(self._a._lay["WS_RB_VELS"])[..., :3])
    @property
    def body_link_ang_vel_w(self): return _proxy(self._a._quad(self._a._lay["WS_RB_VELS"] + 1)[..., :3])
    body_pos_w = body_link_pos_w; body_quat_w = body_link_quat_w; body_pose_w = body_link_pose_w
    body_vel_w = body_link_vel_w; body_lin_vel_w = body_link_lin_vel_w; body_ang_vel_w = body_link_ang_vel_w
    body_com_pos_w = body_link_pos_w; body_com_quat_w = body_link_quat_w; body_com_pose_w = body_link_pose_w
    body_com_vel_w = body_link_vel_w; body_com_lin_vel_w = body_link_lin_vel_w; body_com_ang_vel_w = body_link_ang_vel_w
    @property
    def body_state_w(self): return _proxy(torch.cat([self.body_link_pose_w.torch, self.body_link_vel_w.torch], -1))
    body_link_state_w = body_state_w; body_com_state_w = body_state_w
    @property
    def body_acc_w(self): return _proxy(self._a._vel(self._a._lay["WS_KIN_ACC"]))
    body_com_acc_w = body_acc_w
    @property
    def body_lin_acc_w(self): return _proxy(self._a._quad(self._a._lay["WS_KIN_ACC"])[..., :3])
    body_com_lin_acc_w = body_lin_acc_w
    @property
    def body_ang_acc_w(self): return _proxy(self._a._quad(self._a._lay["WS_KIN_ACC"] + 1)[..., :3])
    @property
    def body_mass(self): return _proxy(self._body_mass)
    @property
    def default_mass(self): return _proxy(self._body_mass)
    @property
    def body_inertia(self): return _proxy(self._body_inertia)
    @property
    def default_inertia(self): return _proxy(self._body_inertia)
    @property
    def body_com_pose_b(self): return _proxy(self._body_com_pose_b)
    @property
    def body_com_pos_b(self): return _proxy(self._body_com_pose_b[..., :3])
    @property
    def body_com_quat_b(self): return _proxy(self._body_com_pose_b[..., 3:])

    def __getattr__(self, name: str) -> Any:
        raise NotImplementedError(f"ArticulationData.{name} is not implemented on the Nexus backend yet")


class Articulation(BaseArticulation):
    """Articulation spawned from MJCF into the Nexus CUDA state, one copy per env."""

    def __init__(self, cfg):
        spawn = cfg.spawn
        if not getattr(spawn, "mjcf_path", ""):
            raise ValueError("Nexus Articulation needs cfg.spawn = NexusMjcfCfg(mjcf_path=..., num_envs=...)")
        self.cfg = cfg
        self._spawn = spawn
        self._is_initialized = False
        st, be = NexusManager.state(), NexusManager.backend()
        NexusManager.ensure_envs(int(spawn.num_envs))
        n_env, auto_floor = int(spawn.num_envs), bool(getattr(spawn, "auto_floor", True))
        if hasattr(st, "insert_mjcf_headless_range"):
            # one parse + hull build for the whole batch (per-env parsing cost ~0.45 s/env on the G1)
            st.insert_mjcf_headless_range(spawn.mjcf_path, 0, n_env, None, auto_floor)
        else:
            for env in range(n_env):
                st.insert_mjcf_headless(spawn.mjcf_path, env, None, auto_floor)
        NexusManager.finalize()
        NexusManager.synchronize()

        self._lay = dict(st.ws_layout())
        self._ws_view = st.links_workspace_cuda()
        self._dof_view = st.dof_state_cuda()
        self._ws = torch.as_tensor(self._ws_view, device=_DEV)      # (L, WS_QUADS, NB, 4)
        self._dof = torch.as_tensor(self._dof_view, device=_DEV)    # (sections, D, NB)
        assert self._ws.data_ptr() == self._ws_view.ptr and self._dof.data_ptr() == self._dof_view.ptr
        L, NB = self._lay["links_per_batch"], self._lay["num_batches"]

        # ---- DOF map from links_static: (rb, parent, mb, assembly_id, ndofs, kinematic, locked, motor_axes)
        stat = torch.as_tensor(st.links_static_host(be)).long()
        asm, ndofs, locked = stat[:, 3], stat[:, 4], stat[:, 6]
        is_root = (stat[:, 1] > L) | (ndofs == 6)
        self._root_link = int(torch.nonzero(is_root).flatten()[0]) if is_root.any() else 0
        self._root_cols = torch.arange(6, device=_DEV) + int(asm[self._root_link])
        joint_links = [k for k in range(L) if ndofs[k] > 0 and not is_root[k]]
        free = [[a for a in range(6) if not (int(locked[k]) >> a) & 1][: int(ndofs[k])] for k in range(L)]
        rows = torch.tensor([k for k in joint_links for _ in range(int(ndofs[k]))])
        slot = torch.tensor([free[k][j] for k in joint_links for j in range(int(ndofs[k]))])
        cols = torch.tensor([int(asm[k]) + j for k in joint_links for j in range(int(ndofs[k]))])
        order = torch.argsort(cols)
        self._rows, self._slot, self._cols = (x[order].to(_DEV) for x in (rows, slot, cols))
        self._quad_of_slot = self._lay["WS_COORDS"] + (self._slot >= 4).long()     # coords 0-3 / 4-5
        self._col_of_slot = torch.where(self._slot >= 4, self._slot - 4, self._slot)
        self._joint_links = [int(rows[order][i]) for i in range(len(order))]
        self._joint_axis = [int(slot[order][i]) for i in range(len(order))]
        self._num_joints = int(len(cols))

        # ---- names (body order == link order; joint order == DOF column order)
        nm = st.mjcf_names(be)
        self._body_names = list(nm["link_body_names"])
        ljn = nm["link_joint_names"]
        self._joint_names = [ljn[k] or f"dof_{i}" for i, k in enumerate(self._joint_links)]

        # ---- engine motors OFF (Python actuator models drive generalized forces)
        for a in sorted(set(self._joint_axis)):
            links = [self._joint_links[i] for i, ax in enumerate(self._joint_axis) if ax == a]
            st.set_motor_gains(be, [int(x) for x in links], int(a), 0.0, 0.0, 0.0)
        self._effort = torch.as_tensor(st.external_gen_forces_cuda(), device=_DEV)   # (D, NB) live
        self._effort[:] = 0.0
        # dof_state sections: 0 velocity, 1 viscous damping, 2 armature (mass-matrix diagonal), 3 Coulomb friction
        self._eng_damping, self._eng_armature, self._eng_friction = self._dof[1], self._dof[2], self._dof[3]

        self._data = ArticulationData(self)
        self._load_mjcf_properties(spawn.mjcf_path)
        self._ALL_JOINT_INDICES = torch.arange(self._num_joints, device=_DEV)
        self._pd_kp = torch.zeros(NB, self._num_joints, device=_DEV); self._pd_kd = torch.zeros_like(self._pd_kp)
        self._pd_ff = torch.zeros_like(self._pd_kp); self._pd_qt = torch.zeros_like(self._pd_kp)
        if os.environ.get("NEXUS_PD_SUBSTEP", "1") == "1":
            NexusManager.post_step_hooks.append(self._pd_substep)
        # Joint velocity limits (`velocity_limit_sim` from the actuator cfgs; PhysX applies them as a hard joint
        # drive limit inside its solver, the engine has no equivalent): clamp the generalized velocities after
        # every physics step. NEXUS_JOINT_VEL_CLAMP=0 disables (diagnostics).
        if os.environ.get("NEXUS_JOINT_VEL_CLAMP", "1") == "1":
            NexusManager.post_step_hooks.append(self._clamp_joint_velocities)
        self._prev_v = torch.zeros(NB, self._num_joints, device=_DEV)
        self._pending = False

        # ---- initial state from cfg.init_state (Isaac semantics) + placement, then template
        self._apply_init_state()
        st.publish_reset_template(be)
        NexusManager.synchronize()

        # ---- actuator models (same construction as isaaclab_physx)
        self.actuators: dict[str, Any] = {}
        self._process_actuators_cfg()
        # ---- wrench composers
        from isaaclab.utils.wrench_composer import WrenchComposer
        self.permanent_wrench_composer = WrenchComposer(self)
        self.instantaneous_wrench_composer = WrenchComposer(self)
        self._is_initialized = True
        NexusManager.register_asset(cfg.prim_path, self)

    # ------------------------------------------------------------------ setup helpers
    def _load_mjcf_properties(self, path: str) -> None:
        """Joint limits, masses and inertias from the MJCF via mujoco (same file the engine loaded)."""
        try:
            import mujoco
        except ImportError:
            warnings.warn("mujoco not available: joint limits / masses stay at defaults"); return
        m = mujoco.MjModel.from_xml_path(path)
        d = self._data
        jl = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(m.njnt)}
        for i, n in enumerate(self._joint_names):
            j = jl.get(n)
            if j is None or not m.jnt_limited[j]:
                continue
            d._joint_pos_limits[:, i, 0] = float(m.jnt_range[j, 0]); d._joint_pos_limits[:, i, 1] = float(m.jnt_range[j, 1])
            a = int(m.jnt_dofadr[j])
            d._joint_friction_coeff[:, i] = float(m.dof_frictionloss[a])
        d._joint_armature.copy_(self._eng_armature[self._cols].T); d._joint_damping.copy_(self._eng_damping[self._cols].T)
        self._eng_friction[self._cols, :] = d._joint_friction_coeff.T
        bl = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b): b for b in range(m.nbody)}
        for i, n in enumerate(self._body_names):
            b = bl.get(n)
            if b is None:
                continue
            d._body_mass[:, i] = float(m.body_mass[b])
            I = m.body_inertia[b]
            d._body_inertia[:, i, 0] = float(I[0]); d._body_inertia[:, i, 4] = float(I[1]); d._body_inertia[:, i, 8] = float(I[2])
            d._body_com_pose_b[:, i, :3] = torch.as_tensor(m.body_ipos[b], dtype=torch.float32, device=_DEV)
        # actuator ctrl ranges as effort limits (overridden per actuator group by cfg)
        for k in range(m.nu):
            j = int(m.actuator_trnid[k, 0]); n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            if n in self._joint_names and m.actuator_ctrllimited[k]:
                d._joint_effort_limits[:, self._joint_names.index(n)] = float(abs(m.actuator_ctrlrange[k]).max())

    def _apply_init_state(self) -> None:
        """Write cfg.init_state (root pos/rot/vel, joint pos/vel by regex) into every env and record defaults."""
        d, N = self._data, self.num_instances
        ist = getattr(self.cfg, "init_state", None)
        jp = torch.zeros(N, self._num_joints, device=_DEV); jv = torch.zeros_like(jp)
        if ist is not None:
            for pat, v in (ist.joint_pos or {}).items():
                ids, _ = _match(self._joint_names, pat); jp[:, ids] = float(v)
            for pat, v in (ist.joint_vel or {}).items():
                ids, _ = _match(self._joint_names, pat); jv[:, ids] = float(v)
            pos = torch.tensor(list(ist.pos), device=_DEV, dtype=torch.float32)
            rot = torch.tensor(list(ist.rot), device=_DEV, dtype=torch.float32)            # (x, y, z, w), Isaac Lab 3.0
            lin = torch.tensor(list(ist.lin_vel), device=_DEV); ang = torch.tensor(list(ist.ang_vel), device=_DEV)
        else:
            pos = torch.zeros(3, device=_DEV); rot = torch.tensor([0.0, 0, 0, 1.0], device=_DEV); lin = ang = torch.zeros(3, device=_DEV)
        origins = getattr(self._spawn, "env_origins", None)
        pose = torch.cat([pos, rot]).expand(N, 7).clone()
        if origins is not None:
            pose[:, :3] += torch.as_tensor(origins, device=_DEV, dtype=torch.float32)
        d._default_joint_pos.copy_(jp); d._default_joint_vel.copy_(jv)
        d._default_root_pose.copy_(pose); d._default_root_vel.copy_(torch.cat([lin, ang]).expand(N, 6))
        self.write_joint_state_to_sim(jp, jv)
        self.write_root_pose_to_sim(pose); self.write_root_velocity_to_sim(d._default_root_vel)
        NexusManager.synchronize()

    def _process_actuators_cfg(self) -> None:
        d = self._data
        for name, acfg in (self.cfg.actuators or {}).items():
            joint_ids, joint_names = self.find_joints(acfg.joint_names_expr)
            if not joint_names:
                raise ValueError(f"No joints found for actuator group {name!r}: {acfg.joint_names_expr}")
            jid = slice(None) if len(joint_names) == self._num_joints else torch.tensor(joint_ids, device=_DEV, dtype=torch.int32)
            act = acfg.class_type(
                cfg=acfg, joint_names=joint_names, joint_ids=jid, num_envs=self.num_instances, device=_DEV,
                stiffness=d._joint_stiffness[:, jid], damping=d._joint_damping[:, jid], armature=d._joint_armature[:, jid],
                friction=d._joint_friction_coeff[:, jid], dynamic_friction=d._joint_dynamic_friction_coeff[:, jid],
                viscous_friction=d._joint_viscous_friction_coeff[:, jid],
                effort_limit=d._joint_effort_limits[:, jid].clone(), velocity_limit=d._joint_vel_limits[:, jid],
            )
            self.actuators[name] = act
            j = self._ALL_JOINT_INDICES if jid == slice(None) else jid.long()
            self._eng_damping[self._cols[j], :] = 0.0
            for src, dst in ((act.effort_limit_sim, d._joint_effort_limits), (act.velocity_limit_sim, d._joint_vel_limits),
                             (act.stiffness, d._joint_stiffness), (act.damping, d._joint_damping), (act.armature, d._joint_armature),
                             (act.friction, d._joint_friction_coeff)):
                if isinstance(src, torch.Tensor):
                    dst[:, j] = src.clone().to(dst.dtype)
                else:
                    dst[:, j] = float(src)
            self._eng_armature[self._cols[j], :] = d._joint_armature[:, j].T
            self._eng_friction[self._cols[j], :] = d._joint_friction_coeff[:, j].T

    # ------------------------------------------------------------------ view helpers
    def _quad(self, q: int) -> torch.Tensor:        # (NB, L, 4)
        return self._ws[:, q, :, :].permute(1, 0, 2)

    def _pose_xyzw(self, f: int) -> torch.Tensor:   # (NB, L, 7) x y z qx qy qz qw
        return torch.cat([self._quad(f + 1)[..., :3], self._quad(f)], dim=-1)

    def _vel(self, f: int) -> torch.Tensor:         # (NB, L, 6)
        return torch.cat([self._quad(f)[..., :3], self._quad(f + 1)[..., :3]], dim=-1)

    def _q_flat(self) -> torch.Tensor:              # (NB, J)
        return self._ws[self._rows, self._quad_of_slot, :, self._col_of_slot].T

    def _pd_substep(self) -> None:
        """Re-evaluate the actuators' PD law at the physics rate (see `_apply_actuator_model`)."""
        d = self._data; q, v = self._q_flat(), self._v_flat()
        tau = self._pd_kp * (self._pd_qt - q) - self._pd_kd * v + self._pd_ff
        lim = d._joint_effort_limits
        tau = torch.where(self._pd_kp > 0, torch.maximum(torch.minimum(tau, lim), -lim), d._applied_torque)
        d._applied_torque[:] = tau; self._effort[self._cols, :] = tau.T
    def _clamp_joint_velocities(self) -> None:
        lim = self._data._joint_vel_limits                        # (NB, J), inf where unset
        if not torch.isfinite(lim).any(): return
        v = self._dof[0]; rows = self._cols                       # velocity section (D, NB); joint rows
        v[rows] = torch.maximum(torch.minimum(v[rows], lim.T), -lim.T)
    def _v_flat(self) -> torch.Tensor:              # (NB, J) pure view of the velocity section
        return self._dof[0][self._cols].T

    # ------------------------------------------------------------------ identity
    @property
    def data(self) -> ArticulationData: return self._data
    @property
    def device(self) -> str: return _DEV
    @property
    def is_initialized(self) -> bool: return self._is_initialized
    @property
    def num_instances(self) -> int: return self._lay["num_batches"]
    @property
    def num_bodies(self) -> int: return self._lay["links_per_batch"]
    @property
    def num_joints(self) -> int: return self._num_joints
    @property
    def num_base_dofs(self) -> int: return 0
    @property
    def is_fixed_base(self) -> bool: return False
    @property
    def num_fixed_tendons(self) -> int: return 0
    @property
    def num_spatial_tendons(self) -> int: return 0
    @property
    def body_names(self) -> list[str]: return self._body_names
    @property
    def joint_names(self) -> list[str]: return self._joint_names
    @property
    def fixed_tendon_names(self) -> list[str]: return []
    @property
    def spatial_tendon_names(self) -> list[str]: return []
    # ---- PhysX-view compatibility for AGILE's fallen-state collection loop ------------------------
    # `FallenStateDataset.collect` bypasses `write_data_to_sim` and, before every raw physics step, does
    #   wp.to_torch(robot._joint_effort_target_sim)[:] = 0.0
    #   robot.root_view.set_dof_actuation_forces(robot._joint_effort_target_sim, robot._ALL_INDICES)
    # On PhysX that zeroes the explicit actuation force while the implicit joint drives keep acting. Here
    # the drives are Python PD models applied in `write_data_to_sim`, so the equivalent is: take the given
    # forces as the explicit effort target and recompute the actuator model from the current state.
    @property
    def _joint_effort_target_sim(self):
        if getattr(self, "_effort_target_sim_wp", None) is None:
            self._effort_target_sim_wp = wp.zeros((self.num_instances, self._num_joints), dtype=wp.float32, device=str(_DEV))
        return self._effort_target_sim_wp
    @property
    def _ALL_INDICES(self): return torch.arange(self.num_instances, device=_DEV)
    @property
    def root_view(self):
        if getattr(self, "_root_view_obj", None) is None: self._root_view_obj = _NexusRootView(self)
        return self._root_view_obj
    @property
    def root_physx_view(self): return self.root_view

    def _link_of_body(self, body_index: int) -> int:
        return int(body_index)

    def find_bodies(self, name_keys, preserve_order: bool = False):
        return _match(self._body_names, name_keys, preserve_order)

    def find_joints(self, name_keys, joint_subset=None, preserve_order: bool = False):
        names = self._joint_names if joint_subset is None else [n for n in self._joint_names if n in joint_subset]
        idx, found = _match(names, name_keys, preserve_order)
        if joint_subset is not None:
            idx = [self._joint_names.index(n) for n in found]
        return idx, found

    def find_fixed_tendons(self, name_keys, tendon_subsets=None, preserve_order=False):
        return [], []

    # ------------------------------------------------------------------ index helpers
    @staticmethod
    def _ids(ids, n) -> torch.Tensor:
        if ids is None or (isinstance(ids, slice) and ids == slice(None)):
            return torch.arange(n, device=_DEV)
        t = _t(ids)
        if t.dtype == torch.bool:
            return torch.nonzero(t).flatten()
        return t.long()

    def _env_ids(self, env_ids=None, env_mask=None) -> torch.Tensor:
        if env_mask is not None:
            return torch.nonzero(_t(env_mask)).flatten()
        return self._ids(env_ids, self.num_instances)

    # ------------------------------------------------------------------ targets (actuator inputs)
    def set_joint_position_target_index(self, target, joint_ids=None, env_ids=None):
        t = _t(target); j = self._ids(joint_ids, self._num_joints); e = self._ids(env_ids, self.num_instances)
        self._data._joint_pos_target[e[:, None], j[None, :]] = t.reshape(len(e), len(j)).to(torch.float32)

    def set_joint_velocity_target_index(self, target, joint_ids=None, env_ids=None):
        t = _t(target); j = self._ids(joint_ids, self._num_joints); e = self._ids(env_ids, self.num_instances)
        self._data._joint_vel_target[e[:, None], j[None, :]] = t.reshape(len(e), len(j)).to(torch.float32)

    def set_joint_effort_target_index(self, target, joint_ids=None, env_ids=None):
        t = _t(target); j = self._ids(joint_ids, self._num_joints); e = self._ids(env_ids, self.num_instances)
        self._data._joint_effort_target[e[:, None], j[None, :]] = t.reshape(len(e), len(j)).to(torch.float32)

    def set_joint_position_target(self, target, joint_ids=None, env_ids=None):
        self.set_joint_position_target_index(target, joint_ids=joint_ids, env_ids=env_ids)
    def set_joint_velocity_target(self, target, joint_ids=None, env_ids=None):
        self.set_joint_velocity_target_index(target, joint_ids=joint_ids, env_ids=env_ids)
    def set_joint_effort_target(self, target, joint_ids=None, env_ids=None):
        self.set_joint_effort_target_index(target, joint_ids=joint_ids, env_ids=env_ids)
    def set_joint_position_target_mask(self, target, joint_mask=None, env_mask=None):
        self.set_joint_position_target_index(target, joint_ids=joint_mask, env_ids=env_mask)
    def set_joint_velocity_target_mask(self, target, joint_mask=None, env_mask=None):
        self.set_joint_velocity_target_index(target, joint_ids=joint_mask, env_ids=env_mask)
    def set_joint_effort_target_mask(self, target, joint_mask=None, env_mask=None):
        self.set_joint_effort_target_index(target, joint_ids=joint_mask, env_ids=env_mask)

    # ------------------------------------------------------------------ actuation + wrenches -> sim
    def _apply_actuator_model(self) -> None:
        d = self._data
        q, v = d.joint_pos.torch, d.joint_vel.torch
        for act in self.actuators.values():
            j = act.joint_indices
            ca = ArticulationActions(joint_positions=d._joint_pos_target[:, j], joint_velocities=d._joint_vel_target[:, j],
                                     joint_efforts=d._joint_effort_target[:, j], joint_indices=j)
            act.compute(ca, joint_pos=q[:, j], joint_vel=v[:, j])
            d._computed_torque[:, j] = act.computed_effort
            lim = d._joint_effort_limits[:, j]                       # effort_limit_sim: PhysX's drive clips to it; be explicit here too
            d._applied_torque[:, j] = torch.maximum(torch.minimum(act.applied_effort, lim), -lim)
            # Effective PD target the actuator model used this control step (delay models etc. included):
            # tau = kp (q_t - q) - kd v + ff  =>  q_t = q + (tau - ff + kd v) / kp. Re-evaluated at every physics
            # substep by `_pd_substep` (PhysX's implicit drive acts at the physics rate; a 20 ms zero-order hold
            # of an explicit torque with AGILE's lightly damped gains rings and whips the limbs).
            kp, kd = act.stiffness, act.damping; ff = d._joint_effort_target[:, j]
            qt = torch.where(kp > 0, q[:, j] + (act.computed_effort - ff + kd * v[:, j]) / torch.where(kp > 0, kp, torch.ones_like(kp)), q[:, j])
            self._pd_kp[:, j], self._pd_kd[:, j], self._pd_ff[:, j], self._pd_qt[:, j] = kp, kd, ff, qt

    def write_data_to_sim(self) -> None:
        self._apply_actuator_model()
        self._effort[self._cols, :] = self._data._applied_torque.T       # zero-copy write into the sim
        # body wrenches -> root free joint (base-DOF projection)
        wr = torch.zeros(self.num_instances, 6, device=_DEV)
        for comp in (self.permanent_wrench_composer, self.instantaneous_wrench_composer):
            if comp.active:
                # The composer's OUTPUT is the total wrench per body in the BODY frame (global + local
                # inputs combined; `set_external_force_and_torque` fills the local buffers by default,
                # so reading the raw global buffers -- as this did before -- saw nothing at all).
                # Rotate to world, then transport every body's wrench to the root: F_root = sum F,
                # tau_root = sum (tau + (p_body_com - p_root) x F).
                # Compose in torch from the input buffers (the composer's own warp kernel wants vec3f-typed
                # views this backend does not provide): world = global + R(body) * local. The torque
                # buffers already carry the moment of any position offsets given at set time.
                q = self._data.body_link_quat_w.torch                                            # (N, B, 4) xyzw
                Fw = (wp.to_torch(comp.global_force_w) + wp.to_torch(comp.global_force_at_com_w)
                      + _quat_apply(q, wp.to_torch(comp.local_force_b)))
                Tw = wp.to_torch(comp.global_torque_w) + _quat_apply(q, wp.to_torch(comp.local_torque_b))
                r = self._data.body_com_pos_w.torch - self._data.root_link_pos_w.torch[:, None, :]
                wr[:, :3] += Fw.sum(1); wr[:, 3:] += (Tw + torch.cross(r, Fw, dim=-1)).sum(1)
        # The engine's free-joint generalized forces are expressed in the ROOT LINK frame (measured:
        # a world +X force at a 180-degree-yawed root produced -vx). Rotate the world wrench into it.
        qr = self._data.root_link_quat_w.torch
        wr = torch.cat([math_utils.quat_apply_inverse(qr, wr[:, :3]), math_utils.quat_apply_inverse(qr, wr[:, 3:])], dim=1)
        self._effort[self._root_cols, :] = wr.T
        self.instantaneous_wrench_composer.reset()

    def update(self, dt: float) -> None:
        NexusManager.synchronize()
        v = self._v_flat()
        if dt > 0:
            self._data._joint_acc.copy_((v - self._prev_v) / dt)
        self._prev_v.copy_(v)

    # ------------------------------------------------------------------ state writes
    def write_root_pose_to_sim(self, root_pose, env_ids=None, env_mask=None) -> None:
        e = self._env_ids(env_ids, env_mask)
        p = _t(root_pose).reshape(len(e), 7).to(torch.float32)
        C, R = self._lay["WS_COORDS"], self._lay["WS_JOINT_ROT"]
        self._ws[self._root_link, C, e, :3] = p[:, :3]
        self._ws[self._root_link, R, e, :] = _wxyz_to_xyzw(p[:, 3:7])
        # keep the pose views coherent before the next FK (readers between write and step)
        self._ws[self._root_link, self._lay["WS_LTW"] + 1, e, :3] = p[:, :3]
        self._ws[self._root_link, self._lay["WS_LTW"], e, :] = _wxyz_to_xyzw(p[:, 3:7])
        self._ws[self._root_link, self._lay["WS_LTP"] + 1, e, :3] = p[:, :3]
        self._ws[self._root_link, self._lay["WS_LTP"], e, :] = _wxyz_to_xyzw(p[:, 3:7])

    def write_root_velocity_to_sim(self, root_velocity, env_ids=None, env_mask=None) -> None:
        e = self._env_ids(env_ids, env_mask)
        vv = _t(root_velocity).reshape(len(e), 6).to(torch.float32)
        self._dof[0][self._root_cols[:, None], e[None, :]] = vv.T
        self._ws[self._root_link, self._lay["WS_RB_VELS"], e, :3] = vv[:, :3]
        self._ws[self._root_link, self._lay["WS_RB_VELS"] + 1, e, :3] = vv[:, 3:]

    def write_root_state_to_sim(self, root_state, env_ids=None, env_mask=None) -> None:
        s = _t(root_state); self.write_root_pose_to_sim(s[:, :7], env_ids, env_mask); self.write_root_velocity_to_sim(s[:, 7:13], env_ids, env_mask)

    write_root_pose_to_sim_index = write_root_pose_to_sim; write_root_velocity_to_sim_index = write_root_velocity_to_sim
    write_root_state_to_sim_index = write_root_state_to_sim
    write_root_link_pose_to_sim = write_root_pose_to_sim; write_root_com_pose_to_sim = write_root_pose_to_sim
    write_root_link_velocity_to_sim = write_root_velocity_to_sim; write_root_com_velocity_to_sim = write_root_velocity_to_sim
    write_root_link_state_to_sim = write_root_state_to_sim; write_root_com_state_to_sim = write_root_state_to_sim
    write_root_link_pose_to_sim_index = write_root_pose_to_sim; write_root_com_pose_to_sim_index = write_root_pose_to_sim
    write_root_link_velocity_to_sim_index = write_root_velocity_to_sim; write_root_com_velocity_to_sim_index = write_root_velocity_to_sim
    def write_root_pose_to_sim_mask(self, root_pose, env_mask=None): self.write_root_pose_to_sim(root_pose, env_mask=env_mask)
    def write_root_velocity_to_sim_mask(self, root_velocity, env_mask=None): self.write_root_velocity_to_sim(root_velocity, env_mask=env_mask)

    def write_joint_position_to_sim(self, position, joint_ids=None, env_ids=None, env_mask=None) -> None:
        e = self._env_ids(env_ids, env_mask); j = self._ids(joint_ids, self._num_joints)
        p = _t(position).reshape(len(e), len(j)).to(torch.float32)
        self._ws[self._rows[j][:, None], self._quad_of_slot[j][:, None], e[None, :], self._col_of_slot[j][:, None]] = p.T

    def write_joint_velocity_to_sim(self, velocity, joint_ids=None, env_ids=None, env_mask=None) -> None:
        e = self._env_ids(env_ids, env_mask); j = self._ids(joint_ids, self._num_joints)
        vv = _t(velocity).reshape(len(e), len(j)).to(torch.float32)
        self._dof[0][self._cols[j][:, None], e[None, :]] = vv.T
        self._prev_v[e[:, None], j[None, :]] = vv

    def write_joint_state_to_sim(self, position, velocity, joint_ids=None, env_ids=None, env_mask=None) -> None:
        self.write_joint_position_to_sim(position, joint_ids, env_ids, env_mask)
        self.write_joint_velocity_to_sim(velocity, joint_ids, env_ids, env_mask)

    write_joint_position_to_sim_index = write_joint_position_to_sim; write_joint_velocity_to_sim_index = write_joint_velocity_to_sim
    write_joint_state_to_sim_index = write_joint_state_to_sim

    # joint / body property writes: recorded (the engine keeps the MJCF values); explicit actuator models read them
    def _store(self, buf, value, joint_ids=None, env_ids=None):
        e = self._ids(env_ids, self.num_instances); j = self._ids(joint_ids, buf.shape[1])
        v = _t(value)
        buf[e[:, None], j[None, :]] = (v.reshape(len(e), len(j)) if v.numel() > 1 else v.reshape(())).to(buf.dtype)

    def write_joint_stiffness_to_sim(self, stiffness, joint_ids=None, env_ids=None, **kw): self._store(self._data._joint_stiffness, stiffness, joint_ids, env_ids)
    def write_joint_damping_to_sim(self, damping, joint_ids=None, env_ids=None, **kw): self._store(self._data._joint_damping, damping, joint_ids, env_ids)
    def write_joint_armature_to_sim(self, armature, joint_ids=None, env_ids=None, **kw):
        self._store(self._data._joint_armature, armature, joint_ids, env_ids); self._eng_armature[self._cols, :] = self._data._joint_armature.T
    def write_joint_friction_coefficient_to_sim(self, joint_friction_coeff, joint_ids=None, env_ids=None, **kw):
        self._store(self._data._joint_friction_coeff, joint_friction_coeff, joint_ids, env_ids); self._eng_friction[self._cols, :] = self._data._joint_friction_coeff.T
    def write_joint_dynamic_friction_coefficient_to_sim(self, joint_dynamic_friction_coeff, joint_ids=None, env_ids=None, **kw): self._store(self._data._joint_dynamic_friction_coeff, joint_dynamic_friction_coeff, joint_ids, env_ids)
    def write_joint_viscous_friction_coefficient_to_sim(self, joint_viscous_friction_coeff, joint_ids=None, env_ids=None, **kw): self._store(self._data._joint_viscous_friction_coeff, joint_viscous_friction_coeff, joint_ids, env_ids)
    def write_joint_effort_limit_to_sim(self, limits, joint_ids=None, env_ids=None, **kw): self._store(self._data._joint_effort_limits, limits, joint_ids, env_ids)
    def write_joint_velocity_limit_to_sim(self, limits, joint_ids=None, env_ids=None, **kw): self._store(self._data._joint_vel_limits, limits, joint_ids, env_ids)
    def write_joint_position_limit_to_sim(self, limits, joint_ids=None, env_ids=None, **kw):
        e = self._ids(env_ids, self.num_instances); j = self._ids(joint_ids, self._num_joints)
        self._data._joint_pos_limits[e[:, None], j[None, :], :] = _t(limits).reshape(len(e), len(j), 2).to(torch.float32)
    write_actuator_stiffness_to_sim = write_joint_stiffness_to_sim; write_actuator_damping_to_sim = write_joint_damping_to_sim
    def set_masses_index(self, masses, body_ids=None, env_ids=None, **kw): self._store(self._data._body_mass, masses, body_ids, env_ids)
    def set_inertias_index(self, inertias, body_ids=None, env_ids=None, **kw):
        e = self._ids(env_ids, self.num_instances); b = self._ids(body_ids, self.num_bodies)
        self._data._body_inertia[e[:, None], b[None, :], :] = _t(inertias).reshape(len(e), len(b), 9).to(torch.float32)
    def set_coms_index(self, coms, body_ids=None, env_ids=None, **kw):
        e = self._ids(env_ids, self.num_instances); b = self._ids(body_ids, self.num_bodies)
        self._data._body_com_pose_b[e[:, None], b[None, :], :] = _t(coms).reshape(len(e), len(b), 7).to(torch.float32)
    set_masses = set_masses_index; set_inertias = set_inertias_index; set_coms = set_coms_index

    def set_external_force_and_torque(self, forces, torques, body_ids=None, env_ids=None, is_global=False, **kw):
        self.permanent_wrench_composer.set_forces_and_torques_index(forces=forces, torques=torques, body_ids=body_ids, env_ids=env_ids, is_global=is_global)

    # ------------------------------------------------------------------ reset
    def reset(self, env_ids=None, env_mask=None) -> None:
        st, be = NexusManager.state(), NexusManager.backend()
        ids = self._env_ids(env_ids, env_mask)
        e = ids.tolist()
        if not e:
            return
        # zero offsets / dof velocities are the binding's default: passing them
        # explicitly marshals len(e) * dofs Python floats per reset
        st.reset_envs(be, e)
        ei = ids.to(_DEV)
        self._effort[:, ei] = 0.0
        d = self._data
        for buf in (d._joint_pos_target, d._joint_vel_target, d._joint_effort_target, d._computed_torque, d._applied_torque, d._joint_acc):
            buf[ei] = 0.0
        self._prev_v[ei] = d._default_joint_vel[ei]
        for act in self.actuators.values():
            act.reset(e)
        self.permanent_wrench_composer.reset(env_ids=ei); self.instantaneous_wrench_composer.reset(env_ids=ei)
        # no synchronize(): the reset is queued on the same stream as the step,
        # and every reader (`update`) already syncs before it touches the views

    # base-class hooks that do not apply here
    def _initialize_impl(self): pass
    def _create_buffers(self): pass
    def _process_cfg(self): pass
    def _process_tendons(self): pass
    def _validate_cfg(self): pass
    def _log_articulation_info(self): pass
    def _set_debug_vis_impl(self, debug_vis): pass
    def _debug_vis_callback(self, event): pass
    def _invalidate_initialize_callback(self, event): pass
    def _initialize_callback(self, event): pass


def _stub(name: str):
    def _missing(self, *args, **kwargs):
        raise NotImplementedError(f"Articulation.{name} is not implemented on the Nexus backend yet")
    _missing.__name__ = name
    return _missing


for _name in sorted(getattr(BaseArticulation, "__abstractmethods__", ())):
    if _name not in Articulation.__dict__:
        setattr(Articulation, _name, _stub(_name))
Articulation.__abstractmethods__ = frozenset()


class _NexusRootView:
    """Minimal stand-in for `ArticulationView` used by AGILE's dataset collection (see `root_view`)."""
    def __init__(self, art): self._art = art
    def set_dof_actuation_forces(self, forces, indices=None):
        a = self._art
        # The collection loop calls this right after a raw `sim.step()`. The engine's step is asynchronous
        # with respect to torch's stream and everything below reads/writes the zero-copy state buffers:
        # without a sync the actuator model sees a half-written state (4096-env collections came out with
        # joint speeds ~50 rad/s and hovering robots; with a per-step readback they were clean).
        from isaaclab_nexus.physics.nexus_manager import NexusManager as _NM
        _NM.synchronize()
        f = wp.to_torch(forces) if isinstance(forces, wp.array) else _t(forces)
        e = a._ids(indices, a.num_instances)
        if f.shape[0] == a.num_instances: f = f[e]
        a._data._joint_effort_target[e] = f.reshape(len(e), a._num_joints).to(torch.float32)
        # Actuator torques only. NOT `write_data_to_sim()`: that would also re-apply the wrench composer's
        # buffers -- AGILE's LiftAction harness force from the last env step (up to 0.9x body weight, upward)
        # -- on every collection step, which on PhysX is never pushed during collection (no write_data_to_sim
        # there). It kept robots hovering near their 2.8 m spawn and exploded joints.
        a._apply_actuator_model(); a._effort[a._cols, :] = a._data._applied_torque.T; a._effort[a._root_cols, :] = 0.0
        # DEVIATION (collection only): the terrain trimesh is a thin surface and the 2 m collection drops hit it
        # at ~6 m/s = 3 cm per 200 Hz step, past the 2 cm contact margin, so limbs tunnel and a resting body is
        # never pushed back out; the recorded "fallen" state then has a leg under the terrain. Cap the root's
        # downward speed during these raw collection steps (NEXUS_COLLECTION_VZ_MAX m/s, 0 disables).
        n = getattr(self, "_calls", 0) + 1; self._calls = n
        if os.environ.get("NEXUS_SHIM_LOG") == "1" and n % 40 == 1:
            from isaaclab_nexus.physics.nexus_manager import NexusManager as _NM
            jv = a.data.joint_vel.torch.abs().max(1).values; rz = a.data.root_link_pos_w.torch[:, 2]; rv = a.data.root_link_vel_w.torch
            rs = _NM._state.rbd_resize_stats() if hasattr(_NM._state, "rbd_resize_stats") else {}
            print(f"[shim] call {n}: root z median {rz.median():.2f} min {rz.min():.2f} max {rz.max():.2f} | envs jv>50: {int((jv > 50).sum())} jv max {jv.max():.0f} | |v| max {rv[:, :3].norm(dim=-1).max():.1f} |w| max {rv[:, 3:].norm(dim=-1).max():.1f} | contacts cap {rs.get('capacity_per_batch')} colors {rs.get('max_colors')}", flush=True)
        vmax = float(os.environ.get("NEXUS_COLLECTION_VZ_MAX", "3.5"))
        if vmax > 0:
            v = a.data.root_link_vel_w.torch if hasattr(a.data, "root_link_vel_w") else a.data.root_vel_w.torch
            fast = torch.nonzero(v[:, 2] < -vmax).flatten()
            if len(fast):
                vv = v[fast].clone(); vv[:, 2] = -vmax; a.write_root_velocity_to_sim(vv, env_ids=fast)
    def set_dof_actuation_forces_mask(self, forces, mask): self.set_dof_actuation_forces(forces, torch.nonzero(_t(mask)).flatten())
