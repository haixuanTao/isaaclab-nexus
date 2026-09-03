"""Nexus-backed ``ContactSensor`` on the engine's built-in contact force sensor.

The engine accumulates, per sensed multibody link, the NORMAL contact impulse over
each step (`contact_sensor_out`, dispatched after the bias solve on the last
substep). Up to ``MAX_CONTACT_SENSORS`` (32) links can be sensed per multibody.

v1 semantics (stated):
  * ``net_forces_w[env, body]`` = normal impulse / dt along world +Z. The engine
    reports the impulse magnitude, not its direction; for AGILE's uses
    (contact booleans via the norm, air/contact time, impact magnitude) that is
    sufficient. Lateral components are zero.
  * history / air-time / contact-time follow Isaac Lab's ContactSensor formulas.
"""

from __future__ import annotations

import re

import torch
import warp as wp

from isaaclab.sensors.contact_sensor.base_contact_sensor import BaseContactSensor
from isaaclab.utils.warp.proxy_array import ProxyArray

from ...physics.nexus_manager import NexusManager


def _proxy(t: torch.Tensor) -> ProxyArray:
    return ProxyArray(wp.from_torch(t.contiguous()))


class ContactSensorData:
    def __init__(self, n: int, b: int, hist: int):
        self._f = torch.zeros(n, b, 3, device="cuda")
        self._hist = torch.zeros(n, max(hist, 1), b, 3, device="cuda")
        self._pos = torch.zeros(n, b, 3, device="cuda")
        self._quat = torch.zeros(n, b, 4, device="cuda"); self._quat[..., 0] = 1.0
        self._vel = torch.zeros(n, b, 3, device="cuda")                       # AGILE patch fields (impact velocity)
        self.velocities_w_history = torch.zeros(n, max(hist, 1), b, 3, device="cuda")
        self.current_air_time = _proxy(torch.zeros(n, b, device="cuda"))
        self.current_contact_time = _proxy(torch.zeros(n, b, device="cuda"))
        self.last_air_time = _proxy(torch.zeros(n, b, device="cuda"))
        self.last_contact_time = _proxy(torch.zeros(n, b, device="cuda"))

    @property
    def net_forces_w(self) -> ProxyArray: return _proxy(self._f)
    @property
    def net_forces_w_history(self) -> ProxyArray: return _proxy(self._hist)
    @property
    def pos_w(self) -> ProxyArray: return _proxy(self._pos)
    @property
    def quat_w(self) -> ProxyArray: return _proxy(self._quat)
    @property
    def velocities_w(self) -> ProxyArray: return _proxy(self._vel)
    @property
    def force_matrix_w(self): return None


class ContactSensor(BaseContactSensor):
    def __init__(self, cfg):
        self.cfg = cfg
        found = NexusManager.find_asset(cfg.prim_path)
        if found is None:
            raise ValueError(f"ContactSensor: no Nexus articulation registered under {cfg.prim_path!r}")
        self._art = found[1]
        body_expr = cfg.prim_path.rstrip("/").split("/")[-1]
        ids, names = self._art.find_bodies(body_expr)
        if not ids:
            raise ValueError(f"ContactSensor: no bodies match {body_expr!r} in {self._art.body_names}")
        st, be = NexusManager.state(), NexusManager.backend()
        accepted = st.set_contact_sensor_links(be, [int(self._art._link_of_body(i)) for i in ids])
        if accepted < len(ids):
            raise ValueError(f"ContactSensor: engine accepted {accepted} of {len(ids)} sensed links (MAX_CONTACT_SENSORS)")
        self._ids, self._names = ids, names
        self._out = torch.as_tensor(st.contact_sensor_out_cuda(), device="cuda")   # (mbs, NB, MAX), zero-copy
        self._n = self._art.num_instances
        self._b = len(ids)
        self._hist_len = int(getattr(cfg, "history_length", 0) or 0)
        self._dt = float(NexusManager.get_physics_dt())
        # The readout is the accumulated normal impulse. In IMPLICIT-Coriolis mode the engine rebuilds
        # contact constraints every substep, so the readout is the LAST substep's impulse and must be
        # scaled by the substep count (measured: sum/weight = 1/iterations). In EXPLICIT mode the
        # constraints are built once per step and the impulse accumulates over the whole step: no
        # scaling. (Same rule Zealot applies: `sensor_inv_dt`.) Scaling in explicit mode over-reported
        # forces 4x at 4 substeps and pinned AGILE's critic observation at its +-25 kN clip.
        from ...physics.nexus_manager import PhysicsManager as _PM
        _implicit = bool(getattr(_PM._cfg, "implicit_coriolis", False))
        self._iters = int(getattr(_PM._cfg, "solver_iterations", 4)) if _implicit else 1
        self._threshold = float(getattr(cfg, "force_threshold", 1.0) or 1.0)
        self._data = ContactSensorData(self._n, self._b, self._hist_len)
        self._data._pos.copy_(self._art.data.body_link_pos_w.torch[:, ids, :])

    # ---- identity ----
    @property
    def num_instances(self) -> int: return self._n
    @property
    def num_sensors(self) -> int: return self._b
    @property
    def body_names(self) -> list[str]: return self._names
    @property
    def contact_view(self): return None
    @property
    def device(self) -> str: return "cuda:0"
    @property
    def is_initialized(self) -> bool: return True
    @property
    def body_ids(self): return self._ids
    def find_bodies(self, name_keys, preserve_order=False):
        import re as _re
        keys = [name_keys] if isinstance(name_keys, str) else list(name_keys)
        idx = [i for i, n in enumerate(self._names) if any(_re.fullmatch(k, n) for k in keys)]
        return idx, [self._names[i] for i in idx]
    @property
    def data(self) -> ContactSensorData: return self._data

    def reset(self, env_ids=None, env_mask=None) -> None:
        e = slice(None) if env_ids is None else torch.as_tensor(env_ids, device="cuda").long()
        d = self._data
        d._f[e] = 0; d._hist[e] = 0; d.velocities_w_history[e] = 0
        for p in (d.current_air_time, d.current_contact_time, d.last_air_time, d.last_contact_time):
            p.torch[e] = 0

    def update(self, dt: float, force_recompute: bool = False) -> None:
        d = self._data
        imp = self._out[0, :, :self._b]                                   # (NB, B) accumulated normal impulse
        d._f[..., 0] = 0; d._f[..., 1] = 0; d._f[..., 2] = imp * self._iters / self._dt
        if self._hist_len > 0:
            d._hist[:, 1:] = d._hist[:, :-1].clone(); d._hist[:, 0] = d._f
        d._pos.copy_(self._art.data.body_link_pos_w.torch[:, self._ids, :])
        d._quat.copy_(self._art.data.body_link_quat_w.torch[:, self._ids, :])
        d._vel.copy_(self._art.data.body_link_lin_vel_w.torch[:, self._ids, :])
        if self._hist_len > 0:
            d.velocities_w_history[:, 1:] = d.velocities_w_history[:, :-1].clone(); d.velocities_w_history[:, 0] = d._vel
        # air / contact time (Isaac Lab semantics)
        in_contact = d._f.norm(dim=-1) > self._threshold
        ct, at = d.current_contact_time.torch, d.current_air_time.torch
        first_contact = in_contact & (ct == 0) & (at > 0)
        first_air = (~in_contact) & (at == 0) & (ct > 0)
        d.last_air_time.torch[first_contact] = at[first_contact]
        d.last_contact_time.torch[first_air] = ct[first_air]
        ct[:] = torch.where(in_contact, ct + dt, torch.zeros_like(ct))
        at[:] = torch.where(in_contact, torch.zeros_like(at), at + dt)

    def compute_first_contact(self, dt: float, abs_tol: float = 1e-8):
        ct = self._data.current_contact_time.torch
        return _proxy(((ct > 0) & (ct <= dt + abs_tol)).clone())

    def compute_first_air(self, dt: float, abs_tol: float = 1e-8):
        at = self._data.current_air_time.torch
        return _proxy(((at > 0) & (at <= dt + abs_tol)).clone())

    # unused base hooks
    def _initialize_impl(self): pass
    def _create_buffers(self): pass
    def _update_buffers_impl(self, env_mask=None): self.update(self._dt)
    def _set_debug_vis_impl(self, debug_vis): pass
    def _debug_vis_callback(self, event): pass
