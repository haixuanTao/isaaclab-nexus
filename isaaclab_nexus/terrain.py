"""Terrain for the Nexus backend: Isaac Lab's generator, no USD.

`isaaclab.terrains.TerrainGenerator` is numpy + trimesh and needs no stage. Every env is
its own Nexus batch in LOCAL coordinates, so instead of one big terrain mesh with envs
placed on it, each env gets the collider of ITS tile (row = curriculum level, col = type),
centred at its origin. Heights for the ray caster come from per-tile GPU height grids.

Engine note: contact manifolds are bounded per (link, mesh) pair and filled in BVH order,
so very fine meshes under a foot sink; the collider is re-rasterized at `collider_res`
(0.25 m) while the height grid keeps `grid_res` (0.05 m).
"""

from __future__ import annotations

import numpy as np
import torch

from .physics.nexus_manager import NexusManager


def _rasterize(tm, sx: float, sy: float, res: float, zmin: float, zmax: float):
    xs = np.arange(-sx / 2, sx / 2 + 1e-6, res); ys = np.arange(-sy / 2, sy / 2 + 1e-6, res)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    org = np.stack([X.ravel(), Y.ravel(), np.full(X.size, zmax + 5.0)], 1)
    loc, ridx, _ = tm.ray.intersects_location(org, np.tile([0, 0, -1.0], (len(org), 1)), multiple_hits=False)
    Z = np.full(X.size, zmin, dtype=np.float32); Z[ridx] = loc[:, 2]
    return xs, ys, Z.reshape(X.shape)


class NexusTerrain:
    """All tiles of a generated terrain, one collider per env for its assigned tile."""

    def __init__(self, terrain_generator_cfg, num_envs: int, tiles: list[tuple[int, int]] | None = None,
                 tile: tuple[int, int] = (0, 0), grid_res: float = 0.05, floor_half: float = 8.0,
                 device: str = "cuda", collider_res: float | None = None, friction: float | None = None):
        import os
        from isaaclab.terrains import TerrainGenerator
        import nexus3d, trimesh

        if collider_res is None:                       # NEXUS_TERRAIN_RES overrides the default
            collider_res = float(os.environ.get("NEXUS_TERRAIN_RES", 0.25))

        self.num_envs = int(num_envs); self.device = device
        self.gen = TerrainGenerator(cfg=terrain_generator_cfg, device=device)
        mesh = self.gen.terrain_mesh
        self.terrain_origins = np.asarray(self.gen.terrain_origins)          # (rows, cols, 3)
        sx, sy = terrain_generator_cfg.size
        V = np.asarray(mesh.vertices, dtype=np.float32); F = np.asarray(mesh.faces, dtype=np.int64)
        zmin, zmax = float(V[:, 2].min()), float(V[:, 2].max())
        tiles = tiles if tiles is not None else [tile] * self.num_envs
        assert len(tiles) == self.num_envs
        uniq = sorted(set(tiles)); self._tile_index = {t: i for i, t in enumerate(uniq)}
        self.tile_of_env = torch.tensor([self._tile_index[t] for t in tiles], device=device)

        st = NexusManager.state(); NexusManager.ensure_envs(self.num_envs)
        heights, colliders = [], []
        self.tile_vertices, self.tile_faces, self.tile_zmin = {}, {}, {}
        for (r, c) in uniq:
            o = self.terrain_origins[r, c]
            lo, hi = o[:2] - np.array([sx, sy]) / 2, o[:2] + np.array([sx, sy]) / 2
            inside = np.all((V[:, :2] >= lo - 1e-6) & (V[:, :2] <= hi + 1e-6), axis=1)
            keep = inside[F].all(axis=1); Fk = F[keep]; used = np.unique(Fk)
            remap = -np.ones(len(V), dtype=np.int64); remap[used] = np.arange(len(used))
            Vt = V[used].copy(); Ft = remap[Fk]; Vt[:, :2] -= o[:2]                      # tile-local XY, world Z
            tm = trimesh.Trimesh(vertices=Vt, faces=Ft, process=False)
            xs, ys, Z = _rasterize(tm, sx, sy, grid_res, zmin, zmax)
            heights.append(torch.as_tensor(Z, device=device))
            if collider_res is not None and collider_res > 0:
                cx, cy, CZ = _rasterize(tm, sx, sy, collider_res, zmin, zmax)
                nxc, nyc = len(cx), len(cy); CX, CY = np.meshgrid(cx, cy, indexing="ij")
                Vc = np.stack([CX.ravel(), CY.ravel(), CZ.ravel()], 1).astype(np.float32)
                a = (np.arange(nxc - 1)[:, None] * nyc + np.arange(nyc - 1)[None, :]).ravel()
                Fc = np.concatenate([np.stack([a, a + 1, a + nyc + 1], 1), np.stack([a, a + nyc + 1, a + nyc], 1)], 0)
                Vt_c, Ft_c = Vc, Fc
            else:
                Vt_c, Ft_c = Vt, Ft
            self.tile_vertices[(r, c)], self.tile_faces[(r, c)] = Vt_c, Ft_c
            self.tile_zmin[(r, c)] = float(Vt[:, 2].min())                                # this tile's own lowest point
            cb = nexus3d.ColliderBuilder.trimesh([tuple(map(float, v)) for v in Vt_c], [tuple(map(int, f)) for f in Ft_c])
            # rapier's default is 0.5; Isaac terrains carry their own material (AGILE: 1.0)
            colliders.append((cb.friction(float(friction)) if friction is not None else cb).build())
        self.grid_res, self.grid_x0, self.grid_y0 = grid_res, float(xs[0]), float(ys[0])
        self.height = torch.stack(heights)                                  # (T, nx, ny)
        self.num_faces = int(len(next(iter(self.tile_faces.values()))))
        # Thick slab: a limb that tunnels through the thin trimesh (fast drops during AGILE's dataset
        # collection) must find the slab's TOP as its nearest face and be ejected up to the tile's lowest
        # point. A 1 m slab left such limbs resting at its mid-plane, 0.5 m under the surface.
        BACK = 10.0
        _fb = nexus3d.ColliderBuilder.cuboid(floor_half, floor_half, BACK) if floor_half > 0 else None
        if _fb is not None and friction is not None:
            _fb = _fb.friction(float(friction))
        floor = _fb.build() if _fb is not None else None
        # Off-tile apron: a flat trimesh at each tile's lowest point over the whole backstop extent. Robots that
        # roll off the 8x8 m tile (AGILE's dataset collection spawns them up to 3 m off-centre) land on it as
        # they do on the tile itself; the deep cuboid slab below is only a last resort (hull-vs-cuboid contact
        # let falling robots sink ~0.8 m into it, hull-vs-triangle does not).
        aprons = {}
        if floor is not None:
            ax = np.arange(-floor_half, floor_half + 1e-6, 1.0, dtype=np.float32); na = len(ax)
            AX, AY = np.meshgrid(ax, ax, indexing="ij"); aa = (np.arange(na - 1)[:, None] * na + np.arange(na - 1)[None, :]).ravel()
            Fa = np.concatenate([np.stack([aa, aa + 1, aa + na + 1], 1), np.stack([aa, aa + na + 1, aa + na], 1)], 0)[:, [0, 2, 1]]
            for t, zt in self.tile_zmin.items():
                Va = np.stack([AX.ravel(), AY.ravel(), np.full(AX.size, zt - 0.005, np.float32)], 1)
                ab = nexus3d.ColliderBuilder.trimesh([tuple(map(float, v)) for v in Va], [tuple(map(int, f)) for f in Fa])
                aprons[t] = (ab.friction(float(friction)) if friction is not None else ab).build()
        for env in range(self.num_envs):
            st.insert_rigid_body_in(env, nexus3d.RigidBodyBuilder.fixed().build(), colliders[int(self.tile_of_env[env])])
            if aprons:
                st.insert_rigid_body_in(env, nexus3d.RigidBodyBuilder.fixed().build(), aprons[uniq[int(self.tile_of_env[env])]])
            if floor is not None:
                # Backstop top flush with THIS tile's lowest point (was the global terrain minimum, i.e. up to the
                # roughest level's pit depth below a flat tile): the trimesh is a thin surface, so a body that gets
                # under it is only caught here, and this bounds the sink to the tile's own height range.
                zb = self.tile_zmin[uniq[int(self.tile_of_env[env])]]
                st.insert_rigid_body_in(env, nexus3d.RigidBodyBuilder.fixed().translation(nexus3d.Vec3(0.0, 0.0, zb - BACK - 1e-3)).build(), floor)
        self.env_origins = torch.zeros(self.num_envs, 3, device=device)
        NexusManager.set_terrain(self)

    def spawn_z(self, margin: float = 0.05) -> float:
        return float(self.height.max().item()) + margin

    def heights_at(self, xy: torch.Tensor, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Bilinear terrain height at tile-local XY points; xy is (N, ..., 2) with N = envs (or env_ids)."""
        H = self.height; T, nx, ny = H.shape
        t = self.tile_of_env if env_ids is None else self.tile_of_env[env_ids]
        t = t.reshape((-1,) + (1,) * (xy.dim() - 2)).expand(xy.shape[:-1])
        u = ((xy[..., 0] - self.grid_x0) / self.grid_res).clamp(0, nx - 1 - 1e-4)
        v = ((xy[..., 1] - self.grid_y0) / self.grid_res).clamp(0, ny - 1 - 1e-4)
        i0, j0 = u.floor().long(), v.floor().long(); fu, fv = u - i0, v - j0
        h00 = H[t, i0, j0]; h10 = H[t, i0 + 1, j0]; h01 = H[t, i0, j0 + 1]; h11 = H[t, i0 + 1, j0 + 1]
        return (h00 * (1 - fu) + h10 * fu) * (1 - fv) + (h01 * (1 - fu) + h11 * fu) * fv


class NexusTerrainImporter:
    """`TerrainImporter` look-alike for `TerrainImporterCfg` (generator terrains) on Nexus.

    Curriculum bookkeeping (`terrain_levels`, `terrain_types`, `update_env_origins`) follows
    Isaac Lab; the tile geometry of an env is fixed at construction, so a level change is
    recorded but the collider does not move until the next construction (stated limitation).
    """

    def __init__(self, cfg, num_envs: int, device: str = "cuda"):
        self.cfg = cfg; self.device = device; self.num_envs = int(num_envs)
        if cfg.terrain_type != "generator" or cfg.terrain_generator is None:
            raise NotImplementedError(f"NexusTerrainImporter: terrain_type={cfg.terrain_type!r} (only 'generator')")
        self.terrain_generator = cfg.terrain_generator
        gcfg = cfg.terrain_generator
        rows, cols = gcfg.num_rows, gcfg.num_cols
        max_init = rows - 1 if cfg.max_init_terrain_level is None else min(cfg.max_init_terrain_level, rows - 1)
        self.max_terrain_level = rows
        self.terrain_levels = torch.randint(0, max_init + 1, (self.num_envs,), device=device)
        self.terrain_types = torch.div(torch.arange(self.num_envs, device=device), (self.num_envs / cols), rounding_mode="floor").long()
        tiles = [(int(r), int(c)) for r, c in zip(self.terrain_levels.tolist(), self.terrain_types.tolist())]
        mat = getattr(cfg, "physics_material", None)
        fric = float(getattr(mat, "static_friction", 1.0)) if mat is not None else None
        self.terrain = NexusTerrain(gcfg, self.num_envs, tiles=tiles, device=device, friction=fric)
        # Isaac's terrain origins are GLOBAL tile positions (metres apart); this backend keeps every env in
        # tile-local coordinates with its tile collider centred at XY=0. Consumers that place states
        # relative to a tile origin (AGILE's fallen-state reset: root_pos_rel + terrain_origins[level, type])
        # must therefore see XY = 0 here, with the generator's Z kept (the tile meshes keep world Z).
        self.terrain_origins = torch.as_tensor(self.terrain.terrain_origins, device=device, dtype=torch.float32).clone()
        self.terrain_origins[..., :2] = 0.0
        self.env_origins = torch.zeros(self.num_envs, 3, device=device)        # envs live in local coordinates
        self._flat_patches = {k: torch.as_tensor(v, device=device) for k, v in getattr(self.terrain.gen, "flat_patches", {}).items()}
        self._level_changes = 0

    @property
    def flat_patches(self): return self._flat_patches
    @property
    def terrain_names(self): return ["terrain"]
    @property
    def physics_material(self): return self.cfg.physics_material
    @property
    def terrain_type(self): return self.cfg.terrain_type
    def heights_at(self, xy, env_ids=None): return self.terrain.heights_at(xy, env_ids)
    def spawn_z(self, margin=0.05): return self.terrain.spawn_z(margin)

    def update_env_origins(self, env_ids: torch.Tensor, move_up: torch.Tensor, move_down: torch.Tensor):
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids] >= self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0))
        self._level_changes += int((move_up | move_down).sum())

    def configure_env_origins(self, origins=None): pass
    def set_debug_vis(self, debug_vis): return False
