"""
gen_my_init_pose.py  --  Build a SOMA-style "holding a box" zero pose ON OUR OWN
skeleton, SYMMETRIC, for use as the route-A (skeleton-native) initialization_pose.

Our mocap frame 0 is a clean T-pose. The SOMA zero pose differs from a T-pose only
in the arms, so this script:

  1. transfers SOMA's own "T-pose -> zero-pose" GLOBAL arm delta onto OUR RIGHT arm
     (the standard side) through the world-alignment A (our Z-up world -> SOMA
     Y-up world), giving a holding-box right arm on our skeleton;
  2. MIRRORS, across the sagittal plane (normal = our shoulder line):
       * the LEFT arm subtree  <- from the RIGHT arm   (right arm is the standard)
       * the RIGHT leg subtree <- from the LEFT leg     (left leg is the standard)
     so the two sides are exactly symmetric;
  3. keeps spine / neck / head / root and the standard sides at our T-pose;
  4. solves each changed joint's local rotation and writes it back in ITS OWN
     channel order (our rig has mixed per-joint orders).

Output: soma_retargeter/configs/sources/mydata/man_pabox_init_pose.bvh   (single frame, OUR skeleton)
Run:    conda run -n soma-retargeter python tools/gen_my_init_pose.py
"""
import os
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import soma_retargeter.assets.bvh as bb

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
_SOMA = os.path.join(_ROOT, "soma_retargeter", "configs", "sources", "soma")

SRC = os.path.join(_ROOT, "my_data", "man_pabox_B_001.bvh")   # our skeleton; frame 0 = T-pose
SOMA_ZERO = os.path.join(_SOMA, "soma_zero_frame0.bvh")        # holding-box reference
SOMA_TPOSE = os.path.join(_SOMA, "soma_tpose_frame0.bvh")      # clean T-pose
# Canonical pipeline location (resolved by io_utils.get_config_file("sources", "mydata", ...)).
OUT = os.path.join(_ROOT, "soma_retargeter", "configs", "sources", "mydata", "man_pabox_init_pose.bvh")

RIGHT_ARM_TRANSFER = ['RightShoulder', 'RightArm', 'RightForeArm']   # standard arm: holding-box
ARM_TGT_ROOT = 'LeftShoulder'    # mirror this subtree from its Right counterpart
LEG_TGT_ROOT = 'RightUpLeg'      # mirror this subtree from its Left counterpart
SRC_UP = np.array([0.0, 0.0, 1.0])
SOMA_UP = np.array([0.0, 1.0, 0.0])


def q2R(q):
    x, y, z, w = q
    s = 2.0 / (x * x + y * y + z * z + w * w)
    return np.array([[1 - s * (y * y + z * z), s * (x * y - z * w),     s * (x * z + y * w)],
                     [s * (x * y + z * w),     1 - s * (x * x + z * z), s * (y * z - x * w)],
                     [s * (x * z - y * w),     s * (y * z + x * w),     1 - s * (x * x + y * y)]])


def globals_RP(sk, anim, fr=0):
    g = np.array(sk.compute_global_transforms(anim.get_local_transforms(fr)))
    R = {sk.joint_names[i]: q2R(g[i, 3:7]) for i in range(sk.num_joints)}
    P = {sk.joint_names[i]: g[i, :3] for i in range(sk.num_joints)}
    return R, P


def local_R(sk, anim, fr=0):
    """Per-joint LOCAL rotation matrices for one frame."""
    lt = np.array(anim.get_local_transforms(fr))
    return {sk.joint_names[i]: q2R(lt[i, 3:7]) for i in range(sk.num_joints)}


def ortho(up, lat):
    e_up = up / np.linalg.norm(up)
    e_lat = lat - (lat @ e_up) * e_up
    e_lat /= np.linalg.norm(e_lat)
    return np.column_stack([e_lat, e_up, np.cross(e_up, e_lat)])


def subtree_names(sk, root_name):
    root = sk.joint_index(root_name)
    out = []
    for i in range(sk.num_joints):
        j = i
        while j != -1:
            if j == root:
                out.append(sk.joint_names[i])
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
    ssk, sanim = bb.load_bvh(SRC)
    zsk, zanim = bb.load_bvh(SOMA_ZERO)
    tsk, tanim = bb.load_bvh(SOMA_TPOSE)

    our_R, our_P = globals_RP(ssk, sanim)        # our T-pose GLOBAL rotations
    our_L = local_R(ssk, sanim)                  # our T-pose LOCAL rotations
    zero_R, _ = globals_RP(zsk, zanim)
    tpose_R, tpose_P = globals_RP(tsk, tanim)

    names = ssk.joint_names
    parent = ssk.parent_indices
    name2i = {n: i for i, n in enumerate(names)}

    # world alignment used ONLY for the right-arm holding-box transfer (our Z-up -> SOMA Y-up)
    A = ortho(SOMA_UP, tpose_P['LeftArm'] - tpose_P['RightArm']) @ ortho(SRC_UP, our_P['LeftArm'] - our_P['RightArm']).T
    # mirroring is done purely in LOCAL space: our rig is X-symmetric (o_left = D.o_right
    # with D = diag(-1,1,1) for every L/R pair), so a symmetric pose satisfies
    # R_left_local = D . R_right_local . D. This is frame-independent and yields exactly
    # equal L/R bone angles (unlike a world-space reflection, which drifts because the
    # bone offsets live in each joint's own local frame, not world).
    D = np.diag([-1.0, 1.0, 1.0])
    print(f"[A] det {np.linalg.det(A):.4f}   [mirror] local D = diag(-1,1,1)")

    new_L = {}     # joints whose LOCAL rotation we overwrite

    # 1) holding-box transfer onto the RIGHT arm (standard side), then read back its LOCAL
    Rg = dict(our_R)
    for nm in RIGHT_ARM_TRANSFER:
        Rg[nm] = A.T @ (zero_R[nm] @ tpose_R[nm].T) @ A @ our_R[nm]
    for nm in RIGHT_ARM_TRANSFER:
        new_L[nm] = Rg[names[parent[name2i[nm]]]].T @ Rg[nm]

    # 2) mirror in LOCAL space: left arm <- right arm,  right leg <- left leg
    for tgt_root, std in ((ARM_TGT_ROOT, ('Left', 'Right')), (LEG_TGT_ROOT, ('Right', 'Left'))):
        for nm in subtree_names(ssk, tgt_root):
            src = nm.replace(std[0], std[1], 1)
            src_local = new_L.get(src, our_L[src])     # holding-box if set, else T-pose
            new_L[nm] = D @ src_local @ D

    written = set(new_L)
    local_R_out = new_L

    # 4) copy our header + frame-0 motion, overwrite ONLY the written joints' rotations
    lines, m, joints = parse_joint_channels(SRC)
    hier = lines[:m]
    vals = [float(x) for x in lines[m + 3].split()]
    pos = 0
    for name, chans in joints:
        if name in written:
            order = ''.join(ch[0].lower() for ch in chans if 'rotation' in ch)
            ang = Rot.from_matrix(local_R_out[name]).as_euler(order.upper(), degrees=True)
            a_by_axis = {ax: a for ax, a in zip(order, ang)}
            for k, ch in enumerate(chans):
                if 'rotation' in ch.lower():
                    vals[pos + k] = a_by_axis[ch[0].lower()]
        pos += len(chans)

    with open(OUT, 'w') as f:
        f.write('\n'.join(hier))
        f.write('\nMOTION\nFrames: 1\nFrame Time: 0.008333\n')
        f.write(' '.join(f'{v:.8g}' for v in vals) + '\n')
    print('wrote', OUT)

    # verify: symmetry + holding-box stance
    osk, oanim = bb.load_bvh(OUT)
    R2, P2 = globals_RP(osk, oanim)
    Dm = np.diag([-1.0, 1.0, 1.0])

    def in_frame(anchor, j):   # position of j expressed in anchor's local frame
        return R2[anchor].T @ (P2[j] - P2[anchor])

    # symmetry measured in the pair's common-ancestor frame (arms: Spine3, legs: Hips)
    worst = 0.0
    pairs = [('Spine3', 'LeftArm', 'RightArm'), ('Spine3', 'LeftForeArm', 'RightForeArm'),
             ('Spine3', 'LeftHand', 'RightHand'), ('Hips', 'LeftFoot', 'RightFoot'),
             ('Hips', 'LeftToeBase', 'RightToeBase')]
    for anchor, a, b in pairs:
        e = np.linalg.norm(in_frame(anchor, a) - Dm @ in_frame(anchor, b))
        print(f"    {a:14s} vs mirror({b:14s}) [{anchor}]: {e * 1000:8.3f} mm")
        worst = max(worst, e)
    print(f"[verify] worst L/R symmetry error: {worst * 1000:.4f} mm")
    for s in ['Left', 'Right']:
        ua = P2[s + 'ForeArm'] - P2[s + 'Arm']; fa = P2[s + 'Hand'] - P2[s + 'ForeArm']
        ua /= np.linalg.norm(ua); fa /= np.linalg.norm(fa)
        bend = np.degrees(np.arccos(np.clip(ua @ fa, -1, 1)))
        print(f"  {s} arm: upper {np.round(ua, 2)} fore {np.round(fa, 2)} elbow {bend:.1f}deg  (Z-up world)")


if __name__ == '__main__':
    main()
