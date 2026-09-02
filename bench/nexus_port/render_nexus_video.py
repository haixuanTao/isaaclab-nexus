"""Render bench/video/nexus_rollout.npz (poses simulated by Nexus) with MuJoCo's offscreen
renderer: the G1's real visual meshes on env 0's actual terrain tile, 4 envs in a 2x2 grid."""
import os, re, numpy as np, mujoco, trimesh, cv2, subprocess
D = "/workspace/bench/video"; SRC = "/workspace/unitree_mujoco/unitree_robots/g1/g1_29dof.xml"
z = np.load(f"{D}/nexus_rollout.npz", allow_pickle=True)
rp, rq, jp, names = z["root_pos"], z["root_quat"], z["joint_pos"], list(z["joint_names"])
T, K = rp.shape[:2]; fps = round(1.0 / float(z["dt"]))
tm = trimesh.Trimesh(z["terrain_v"], z["terrain_f"], process=False)
if tm.face_normals[:, 2].mean() < 0: tm.invert()                                  # MuJoCo back-face culls; normals must point up
tm.export(f"{D}/terrain_env0.stl")
xml = open(SRC).read()
xml = xml.replace('meshdir="meshes"', f'meshdir="{os.path.dirname(SRC)}/meshes"')
xml = xml.replace("</asset>", f'<mesh name="terrain_tile" file="{D}/terrain_env0.stl"/><texture type="skybox" builtin="gradient" rgb1=".55 .7 .9" rgb2=".95 .97 1" width="256" height="256"/></asset>', 1)
xml = xml.replace("<worldbody>", '<visual><headlight ambient=".45 .45 .45" diffuse=".6 .6 .6"/><global offwidth="1920" offheight="1080"/></visual><worldbody>', 1)
xml = xml.replace("<worldbody>", '<worldbody><light pos="0 0 6" dir="0 0 -1" diffuse=".9 .9 .9"/><light pos="3 -3 4" dir="-.5 .5 -1" diffuse=".5 .5 .5"/>'
                  '<geom type="mesh" mesh="terrain_tile" rgba=".58 .55 .5 1" contype="0" conaffinity="0"/>', 1)
m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
qadr = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): int(m.jnt_qposadr[j]) for j in range(m.njnt)}
free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]; fa = int(m.jnt_qposadr[free[0]])
jidx = [qadr[n] for n in names]
W, H = 480, 360; r = mujoco.Renderer(m, H, W); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance, cam.azimuth, cam.elevation = 2.6, 140, -18
opt = mujoco.MjvOption(); opt.geomgroup[:] = 0; opt.geomgroup[1] = 1; opt.geomgroup[0] = 1     # visual + terrain
raw = f"{D}/nexus_g1_standup_raw.mp4"; vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * W, 2 * H))
for t in range(T):
    tiles = []
    for k in range(K):
        d.qpos[fa:fa + 3] = rp[t, k]; d.qpos[fa + 3:fa + 7] = rq[t, k]           # (w,x,y,z) = MuJoCo order
        d.qpos[jidx] = jp[t, k]; mujoco.mj_forward(m, d)
        cam.lookat[:] = rp[t, k] + np.array([0, 0, 0.2]); r.update_scene(d, cam, opt)
        img = r.render()[:, :, ::-1].copy()
        cv2.putText(img, f"Nexus {os.environ.get('NEXUS_VIDEO_TAG', '')} | env {k} | t={t/fps:4.1f}s", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(img)
    vw.write(np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])]))
vw.release()
out = f"{D}/nexus_g1_standup_{os.environ.get('NEXUS_VIDEO_TAG', 'model99')}.mp4"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out], check=True); os.remove(raw)
print(f"wrote {out}: {T} frames @ {fps} fps, {K} envs, {os.path.getsize(out)/1e6:.1f} MB | GL={os.environ.get('MUJOCO_GL')}")
