"""
gen_mydata2_init_pose.py  --  Build a SOMA-style symmetric "holding a box" zero
pose ON THE MYDATA2 SKELETON, for use as the mydata2 pipeline initialization_pose.

mydata2 is a standard Y-up centimetre BVH (51 joints, spine = Hips->Spine->Spine1,
no Spine2/Spine3, single Neck). This differs from the mydata (man_pabox) skeleton,
so we cannot reuse mydata's init pose file directly. Instead we construct the same
clean stance directly on the mydata2 rig:

  * spine / neck / head / shoulders leveled (rest pose)
  * upper arms vertical, pointing at the ground   (world -Y)
  * forearms horizontal, pointing forward          (world +Z), 90-deg elbow
  * both palms vertical and facing each other       (palm normals along +/-X)
  * fingers straight, thumbs up
  * LEGS + FEET copied from frame 0 of the reference BVH (so grounding matches
    the motion clips); root height aligned to that frame's ankle height

The right side is computed from the rig geometry; the left side is mirrored
(local D = diag(-1, 1, 1)) so the two sides are exactly symmetric.

Y-up axes for this rig:  up = +Y,  down = -Y,  forward = +Z (toes),  lateral = X.

Output: soma_retargeter/configs/sources/mydata2/init_pose.bvh   (single frame)
Run:    python tools/gen_mydata2_init_pose.py \
            [--src dataset/my_data2/Roundhouse_kick_take_001_B.bvh]
"""
import argparse
import os
import numpy as np
from scipy.spatial.transform import Rotation as Rot

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
_DEFAULT_SRC = os.path.join(_ROOT, "dataset", "my_data2", "Roundhouse_kick_take_001_B.bvh")
_OUT = os.path.join(_ROOT, "soma_retargeter", "configs", "sources", "mydata2", "init_pose.bvh")

DOWN = np.array([0.0, -1.0, 0.0])     # upper arm points at the ground
FORWARD = np.array([0.0, 0.0, 1.0])   # forearm / fingers point forward
UP = np.array([0.0, 1.0, 0.0])        # world up (for thumb-up disambiguation)


def sanitize(hier_lines):
    """Fix known corruption in the shipped mydata2 hierarchy."""
    out = []
    for l in hier_lines:
        if "Yrotations" in l:
            l = l.replace("Yrotations", "Yrotation")
        if l.strip() == "}a":
            l = l[: l.index("}") + 1]
        out.append(l)
    return out


def parse(path):
    lines = open(path).read().splitlines()
    m = next(i for i, l in enumerate(lines) if l.strip() == "MOTION")
    hier = sanitize(lines[:m])
    joints, offsets, stack = [], {}, []
    cur = pending = None
    ignore = False
    ends = []
    for l in hier:
        t = l.split()
        if not t:
            continue
        if t[0] in ("ROOT", "JOINT"):
            joints.append([t[1], stack[-1] if stack else None, None])
            cur = pending = t[1]
        elif t[0] == "End" and len(t) > 1 and t[1] == "Site":
            ignore = True
            pending = ("END", cur)
        elif t[0] == "OFFSET":
            off = np.array([float(x) for x in t[1:4]])
            if ignore:
                ends.append((pending[1], off))
                ignore = False
            else:
                offsets[pending] = off
        elif t[0] == "CHANNELS":
            for j in joints:
                if j[0] == cur:
                    j[2] = t[2:]
        elif t[0] == "{":
            stack.append(cur)
        elif t[0] == "}":
            if stack:
                stack.pop()
    return hier, joints, offsets, ends


def order(chans):
    return "".join(c[0].upper() for c in chans if "rotation" in c.lower())


def read_frame0(path, joints, chans):
    """Read frame-0 local rotation matrices + root translation from a BVH."""
    lines = open(path).read().splitlines()
    m = next(i for i, l in enumerate(lines) if l.strip() == "MOTION")
    vals = [float(x) for x in lines[m + 3].split()]
    L, root_t, pos = {}, np.zeros(3), 0
    for name, _, _ in joints:
        cs = chans[name]
        if name == joints[0][0]:
            for k, ch in enumerate(cs):
                cl = ch.lower()
                if cl == "xposition":
                    root_t[0] = vals[pos + k]
                elif cl == "yposition":
                    root_t[1] = vals[pos + k]
                elif cl == "zposition":
                    root_t[2] = vals[pos + k]
        o = order(cs)
        ang = [vals[pos + k] for k, ch in enumerate(cs) if "rotation" in ch.lower()]
        L[name] = Rot.from_euler(o, ang, degrees=True).as_matrix() if o else np.eye(3)
        pos += len(cs)
    return L, root_t


def fk(joints, offsets, L):
    G, P = {}, {}
    for name, parent, _ in joints:
        if parent is None:
            G[name] = L[name]
            P[name] = np.zeros(3)
        else:
            G[name] = G[parent] @ L[name]
            P[name] = P[parent] + G[parent] @ offsets[name]
    return G, P


def swing(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return Rot.align_vectors([b], [a])[0].as_matrix()


def frame(e1, e2):
    e1 = e1 / np.linalg.norm(e1)
    e2 = e2 - (e2 @ e1) * e1
    e2 = e2 / np.linalg.norm(e2)
    return np.column_stack([e1, e2, np.cross(e1, e2)])


def main(src_path=None):
    src = src_path or _DEFAULT_SRC
    if not os.path.isfile(src):
        raise FileNotFoundError(f"Reference BVH not found: {src}")
    print(f"[INFO]: Building mydata2 zero pose from [{src}]")

    hier, joints, offsets, ends = parse(src)
    names = [j[0] for j in joints]
    parent = {j[0]: j[1] for j in joints}
    chans = {j[0]: j[2] for j in joints}
    D = np.diag([-1.0, 1.0, 1.0])
    cap_L, cap_root = read_frame0(src, joints, chans)

    # Upper body (root/spine/neck/head/shoulders) starts leveled at identity; the
    # LEGS + FEET are taken from the data's own standing frame so the zero pose
    # grounds exactly like the motion clips. Building the legs synthetically
    # (straight rest pose) made the character stand ~2 cm lower with non-flat
    # feet, which showed up as a small foot lift after retargeting.
    L = {n: np.eye(3) for n in names}
    for chain in ("UpLeg", "Leg", "Foot", "ToeBase"):
        for s in ("Left", "Right"):
            L[s + chain] = cap_L[s + chain]

    side = "Right"
    other = "Left"

    # 1) upper arm: rest bone dir (+/-X) -> straight down (-Y)
    G_rest, P_rest = fk(joints, offsets, L)
    uA = P_rest[side + "ForeArm"] - P_rest[side + "Arm"]
    G_arm = swing(uA, DOWN)
    L[side + "Arm"] = np.linalg.inv(G_rest[parent[side + "Arm"]]) @ G_arm

    # 2) forearm: 90-deg elbow bend that carries the arm frame down->forward (+Z)
    Rbend = Rot.from_euler("X", -90, degrees=True).as_matrix()   # world: -Y -> +Z
    G_fore = Rbend @ G_arm
    L[side + "ForeArm"] = np.linalg.inv(G_arm) @ G_fore

    # 3) hand: fingers +Z, palm vertical facing center, thumb up
    L[side + "Hand"] = np.eye(3)                    # provisional (continues forearm)
    G, P = fk(joints, offsets, L)
    a = P[side + "HandMiddle1"] - P[side + "Hand"]         # finger dir (~ +Z)
    palm = np.cross(P[side + "HandIndex1"] - P[side + "HandPinky1"], a)  # outward palm
    S = frame(a, palm)
    best = None
    for tx in (np.array([1.0, 0, 0]), np.array([-1.0, 0, 0])):
        Rdelta = frame(FORWARD, tx) @ S.T
        G_hand = Rdelta @ G[side + "Hand"]
        L[side + "Hand"] = np.linalg.inv(G_fore) @ G_hand
        _, Pt = fk(joints, offsets, L)
        thumb = Pt[side + "HandThumb3"] - Pt[side + "Hand"]
        if best is None or (thumb @ UP) > best[0]:
            best = ((thumb @ UP), L[side + "Hand"].copy())
    L[side + "Hand"] = best[1]

    # 4) mirror the standard side onto the other side (exact L/R symmetry)
    for j in ("Arm", "ForeArm", "Hand"):
        L[other + j] = D @ L[side + j] @ D

    # 5) ground the character so the ankles sit at the SAME height as the data's
    #    standing frame (matches the motion => no residual foot lift). Root is
    #    centred in X/Z; height is chosen to align the mean ankle with the data.
    _, P_cap = fk(joints, offsets, cap_L)
    data_ankle_y = float(np.mean([P_cap["LeftFoot"][1], P_cap["RightFoot"][1]]) + cap_root[1])
    _, P = fk(joints, offsets, L)
    my_ankle_y = float(np.mean([P["LeftFoot"][1], P["RightFoot"][1]]))
    root_t = np.array([0.0, data_ankle_y - my_ankle_y, 0.0])

    # ---- write single-frame BVH ----
    vals = []
    for name, _, _ in joints:
        cs = chans[name]
        row = [0.0] * len(cs)
        if name == names[0]:                       # root: fill translation channels
            for k, ch in enumerate(cs):
                cl = ch.lower()
                if cl == "xposition":
                    row[k] = root_t[0]
                elif cl == "yposition":
                    row[k] = root_t[1]
                elif cl == "zposition":
                    row[k] = root_t[2]
        o = order(cs)
        if o:
            ang = Rot.from_matrix(L[name]).as_euler(o, degrees=True)
            by_axis = {ax.lower(): v for ax, v in zip(o, ang)}
            for k, ch in enumerate(cs):
                if "rotation" in ch.lower():
                    row[k] = by_axis[ch[0].lower()]
        vals.extend(row)

    with open(_OUT, "w") as f:
        f.write("\n".join(hier))
        f.write("\nMOTION\nFrames: 1\nFrame Time: 0.005556\n")
        f.write(" ".join(f"{v:.6g}" for v in vals) + "\n")

    _verify(joints, offsets, ends)


def _verify(joints, offsets, ends):
    hier, joints2, offsets2, ends2 = parse(_OUT)
    names = [j[0] for j in joints2]
    L = {}
    # read back the written frame
    lines = open(_OUT).read().splitlines()
    m = next(i for i, l in enumerate(lines) if l.strip() == "MOTION")
    vals = [float(x) for x in lines[m + 3].split()]
    chans = {j[0]: j[2] for j in joints2}
    pos = 0
    for name, _, _ in joints2:
        cs = chans[name]
        o = "".join(c[0].upper() for c in cs if "rotation" in c.lower())
        ang = [vals[pos + k] for k, ch in enumerate(cs) if "rotation" in ch.lower()]
        L[name] = Rot.from_euler(o, ang, degrees=True).as_matrix() if o else np.eye(3)
        pos += len(cs)
    G, P = fk(joints2, offsets2, L)
    ys = [P[n][1] for n in names] + [P[pj][1] + off[1] for pj, off in ends2]
    print(f"[verify] standing height {max(ys) - min(ys):.2f} cm, feet at y={min(ys):.3f}")
    for s in ("Right", "Left"):
        u = P[s + "ForeArm"] - P[s + "Arm"]; u /= np.linalg.norm(u)
        fdir = P[s + "Hand"] - P[s + "ForeArm"]; fdir /= np.linalg.norm(fdir)
        thumb = P[s + "HandThumb3"] - P[s + "Hand"]; thumb /= np.linalg.norm(thumb)
        print(f"  {s:5s} upperarm {np.round(u,2)} forearm {np.round(fdir,2)} thumb {np.round(thumb,2)}")
    print(f"[OK] wrote {_OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Build mydata2 holding-box zero pose from a reference BVH frame 0.")
    ap.add_argument(
        "--src",
        default=_DEFAULT_SRC,
        help="Reference BVH (hierarchy + standing legs from frame 0). "
             f"Default: {_DEFAULT_SRC}")
    args = ap.parse_args()
    main(args.src)
