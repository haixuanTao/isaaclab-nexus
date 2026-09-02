"""Nexus-backed ``RayCaster`` (height scanner).

v1: rays are cast against the static terrain height grid built by
``isaaclab_nexus.terrain.NexusTerrain`` (bilinear lookup), which is exact for
heightfield terrains and needs no GPU ray query. Pattern: ``GridPatternCfg``
(resolution, size, ordering); alignment ``"yaw"`` / ``"base"`` / ``"world"``.
AGILE's height sensor is ``size=(0, 0)`` -> a single ray under the pelvis.
"""

from __future__ import annotations

import math

import torch
import warp as wp

from isaaclab.sensors.ray_caster.base_ray_caster import BaseRayCaster
from isaaclab.utils.warp.proxy_array import ProxyArray

from ...physics.nexus_manager import NexusManager


def _proxy(t: torch.Tensor) -> ProxyArray:
    return ProxyArray(wp.from_torch(t.contiguous()))


class RayCasterData:
    def __init__(self, n: int, r: int):
        self._pos = torch.zeros(n, 3, device="cuda")
        self._quat = torch.zeros(n, 4, device="cuda"); self._quat[:, 0] = 1.0   # w x y z (Isaac convention)
        self._hits = torch.zeros(n, r, 3, device="cuda")

    @property
    def pos_w(self) -> ProxyArray: return _proxy(self._pos)
    @property
    def quat_w(self) -> ProxyArray: return _proxy(self._quat)
    @property
    def ray_hits_w(self) -> ProxyArray: return _proxy(self._hits)


class RayCaster(BaseRayCaster):
    def __init__(self, cfg):
        self.cfg = cfg
        found = NexusManager.find_asset(cfg.prim_path)
        if found is None:
            raise ValueError(f"RayCaster: no Nexus articulation registered under {cfg.prim_path!r}")
        self._art = found[1]
        body = cfg.prim_path.rstrip("/").split("/")[-1]
        ids, _ = self._art.find_bodies(body)
        if not ids:
            raise ValueError(f"RayCaster: body {body!r} not in {self._art.body_names}")
        self._body = ids[0]
        self._terrain = NexusManager.terrain()
        off = getattr(cfg.offset, "pos", (0.0, 0.0, 0.0)) if getattr(cfg, "offset", None) is not None else (0.0, 0.0, 0.0)
        self._offset = torch.tensor(off, device="cuda", dtype=torch.float32)
        pat = cfg.pattern_cfg
        res, (sx, sy) = float(pat.resolution), pat.size
        xs = torch.arange(-sx / 2, sx / 2 + 1e-6, res) if sx > 0 else torch.zeros(1)
        ys = torch.arange(-sy / 2, sy / 2 + 1e-6, res) if sy > 0 else torch.zeros(1)
        ordering = getattr(pat, "ordering", "xy")
        X, Y = torch.meshgrid(xs, ys, indexing="ij" if ordering == "xy" else "xy")
        self._pattern = torch.stack([X.reshape(-1), Y.reshape(-1)], -1).cuda()   # (R, 2) sensor-frame offsets
        self._align = getattr(cfg, "ray_alignment", "yaw")
        self._n = self._art.num_instances
        self._r = int(self._pattern.shape[0])
        self._data = RayCasterData(self._n, self._r)
        self.update(0.0)

    @property
    def num_instances(self) -> int: return self._n
    @property
    def num_rays(self) -> int: return self._r
    @property
    def data(self) -> RayCasterData: return self._data

    def reset(self, env_ids=None, env_mask=None) -> None:
        self.update(0.0)

    def update(self, dt: float, force_recompute: bool = False) -> None:
        d = self._art.data
        pos = d.body_link_pos_w.torch[:, self._body, :]          # (N, 3) world
        quat = d.body_link_quat_w.torch[:, self._body, :]        # (N, 4) x y z w
        self._data._pos.copy_(pos + self._offset)
        self._data._quat[:, 0] = quat[:, 3]; self._data._quat[:, 1:] = quat[:, :3]
        if self._align == "yaw":
            x, y, z, w = quat.unbind(-1)
            yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            c, s = torch.cos(yaw), torch.sin(yaw)
            px, py = self._pattern[:, 0], self._pattern[:, 1]
            ox = c[:, None] * px[None] - s[:, None] * py[None]
            oy = s[:, None] * px[None] + c[:, None] * py[None]
        else:                                                        # "world" (base alignment treated as world in v1)
            ox = self._pattern[:, 0][None].expand(self._n, -1); oy = self._pattern[:, 1][None].expand(self._n, -1)
        hx = pos[:, 0:1] + self._offset[0] + ox
        hy = pos[:, 1:2] + self._offset[1] + oy
        if self._terrain is not None:
            hz = self._terrain.heights_at(torch.stack([hx, hy], -1))
        else:
            hz = torch.zeros_like(hx)
        self._data._hits[..., 0] = hx; self._data._hits[..., 1] = hy; self._data._hits[..., 2] = hz

    # unused base hooks
    def _initialize_impl(self): pass
    def _initialize_pose_tracking(self): pass
    def _initialize_warp_meshes(self): pass
    def _initialize_rays_impl(self): pass
    def _update_buffers_impl(self, env_mask=None): self.update(0.0)
    def _set_debug_vis_impl(self, debug_vis): pass
    def _debug_vis_callback(self, event): pass
