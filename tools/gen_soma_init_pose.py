"""
gen_init_pose.py  --  Build a clean, SYMMETRIC, SOMA-skeleton "holding a box"
initialization pose (the pipeline's initialization_pose / zero pose).

SOMA's stock zero pose (configs/sources/soma/soma_zero_frame0.bvh) is the right stance
(standing, arms forward, elbows bent ~75 deg) but is slightly ASYMMETRIC, and the
GUI / pipeline only accept the SOMA skeleton (the viewer skins the SOMA mesh by
joint name). This script takes soma_zero, mirrors it across the sagittal plane so
the two sides match exactly, and writes it back in the SAME SOMA format so it
loads in `app/bvh_to_csv_converter.py`.

Standard sides (kept as-is, the other side is mirrored from them):
  * ARMS: RIGHT arm is the standard -> LEFT arm chain mirrored from it.
  * LEGS: LEFT  leg is the standard -> RIGHT leg chain mirrored from it.
Center joints (root/hips/spine/neck/head) are left untouched.

Mirroring is done in WORLD space (reflect global transform across the sagittal
plane, normal = shoulder line), then solved back to local — this is robust to the
per-joint local-frame differences in the SOMA rig (e.g. shoulder offsets differ
in Z while arm offsets differ in X).

Output: my_data/man_pabox_init_pose.bvh   (single frame, SOMA skeleton)
Run:    conda run -n soma-retargeter python tools/gen_init_pose.py
"""
import os
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import soma_retargeter.assets.bvh as bb

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
SRC = os.path.join(_ROOT, "soma_retargeter", "configs", "sources", "soma", "soma_zero_frame0.bvh")
OUT = os.path.join(_ROOT, "my_data", "man_pabox_init_pose.bvh")

ARM_STD, ARM_TGT = "Right", "Left"     # right arm is the standard, mirror to left
LEG_STD, LEG_TGT = "Left", "Right"     # left  leg is the standard, mirror to right
ARM_ROOT = "Shoulder"                  # subtree root suffix for the arm
LEG_ROOT = "Leg"                       # subtree root suffix for the leg


def q2R(q):
    x, y, z, w = q
    s = 2.0 / (x * x + y * y + z * z + w * w)
    return np.array([[1 - s * (y * y + z * z), s * (x * y - z * w),     s * (x * z + y * w)],
                     [s * (x * y + z * w),     1 - s * (x * x + z * z), s * (y * z - x * w)],
                     [s * (x * z - y * w),     s * (y * z + x * w),     1 - s * (x * x + y * y)]])


def subtree(sk, root_name):
    """All joint indices in the subtree rooted at root_name (inclusive)."""
    root = sk.joint_index(root_name)
    out = {root}
    for i in range(sk.num_joints):
        j, chain = i, []
        while j != -1:
            chain.append(j)
            if j in out:
                out.update(chain)
                break
            j = sk.parent_indices[j]
    return out


def parse_joint_channels(path):
    lines = open(path).read().splitlines()
    m = next(i for i, l in enumerate(lines) if l.strip() == 'MOTION')
    joints, cur = [], None
    for l in lines[:m]:
        t = l.split()
        if not t:
            continue
        if t[0] in ('ROOT', 'JOINT'):
            cur = t[1]
        elif t[0] == 'CHANNELS' and cur is not None:
            joints.append((cur, t[2:]))
            cur = None
    return lines, m, joints


def main():
    sk, anim = bb.load_bvh(SRC)
    g = np.array(sk.compute_global_transforms(anim.get_local_transforms(0)))
    R = [q2R(g[i, 3:7]) for i in range(sk.num_joints)]
    P = [g[i, :3].copy() for i in range(sk.num_joints)]
    names = sk.joint_names
    parent = sk.parent_indices
    name2i = {n: i for i, n in enumerate(names)}

    # sagittal-plane reflection (normal = world shoulder line)
    n = P[name2i['LeftArm']] - P[name2i['RightArm']]
    n = n / np.linalg.norm(n)
    S = np.eye(3) - 2.0 * np.outer(n, n)
    print(f"[mirror] plane normal {np.round(n, 3)}")

    # which joints get overwritten, and their mirror-source name
    arm_targets = subtree(sk, ARM_TGT + ARM_ROOT)
    leg_targets = subtree(sk, LEG_TGT + LEG_ROOT)
    src_name = {}
    for i in arm_targets:
        src_name[i] = name2i[names[i].replace(ARM_TGT, ARM_STD, 1)]
    for i in leg_targets:
        src_name[i] = name2i[names[i].replace(LEG_TGT, LEG_STD, 1)]
    targets = set(src_name)

    # mirror global transforms of target joints from their standard-side source
    Rn = [Ri.copy() for Ri in R]
    Pn = [Pi.copy() for Pi in P]
    for i in range(sk.num_joints):       # DFS order -> parents before children
        if i in targets:
            s = src_name[i]
            Rn[i] = S @ R[s] @ S
            Pn[i] = S @ P[s]

    # solve local transform (rot + trans) for target joints from mirrored globals
    local_R, local_t = {}, {}
    for i in targets:
        p = parent[i]
        local_R[i] = Rn[p].T @ Rn[i]
        local_t[i] = Rn[p].T @ (Pn[i] - Pn[p])      # meters

    # write: copy SOMA header + frame-0 motion, overwrite ONLY target joints' channels
    lines, m, joints = parse_joint_channels(SRC)
    hier = lines[:m]
    vals = [float(x) for x in lines[m + 3].split()]
    pos = 0
    for name, chans in joints:
        i = name2i[name]
        if i in targets:
            order = ''.join(ch[0].lower() for ch in chans if 'rotation' in ch)
            ang = Rot.from_matrix(local_R[i]).as_euler(order.upper(), degrees=True)
            a_by_axis = {ax: a for ax, a in zip(order, ang)}
            t_cm = local_t[i] * 100.0
            for k, ch in enumerate(chans):
                c = ch.lower()
                if 'xposition' in c:
                    vals[pos + k] = t_cm[0]
                elif 'yposition' in c:
                    vals[pos + k] = t_cm[1]
                elif 'zposition' in c:
                    vals[pos + k] = t_cm[2]
                elif 'rotation' in c:
                    vals[pos + k] = a_by_axis[ch[0].lower()]
        pos += len(chans)

    with open(OUT, 'w') as f:
        f.write('\n'.join(hier))
        f.write('\nMOTION\nFrames: 1\nFrame Time: 0.008333\n')
        f.write(' '.join(f'{v:.8g}' for v in vals) + '\n')
    print('wrote', OUT)

    # verify symmetry: reload, FK, check target == reflection of standard
    osk, oanim = bb.load_bvh(OUT)
    og = np.array(osk.compute_global_transforms(oanim.get_local_transforms(0)))
    worst = 0.0
    for i, s in src_name.items():
        err = np.linalg.norm(og[i, :3] - S @ og[s, :3])
        worst = max(worst, err)
    print(f"[verify] worst target-vs-mirrored-standard position error: {worst * 1000:.4f} mm")


if __name__ == '__main__':
    main()
