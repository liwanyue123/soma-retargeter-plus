"""
gen_mydata3_init_pose.py  --  Build a SOMA-style symmetric "holding a box" zero
pose ON THE MYDATA3 SKELETON, for use as the mydata3 pipeline initialization_pose.

mydata3 is a standard Y-up centimetre BVH (21 joints, spine = Hips->Spine->Spine1,
single Neck, NO finger chains) used by clips like dataset/my_data3/PM_dance_002_han.bvh.
The joint names are a strict subset of mydata2's, but the actor's proportions differ
substantially (shoulders/torso ~1.4x wider relative to the legs), so it gets its own
source type + init pose instead of reusing mydata2's.

Same stance as gen_mydata2_init_pose.py:

  * spine / neck / head straight (rest pose, all rotations zero)
  * legs + feet taken from the data's own standing frame (frame 0) so the zero
    pose grounds exactly like the motion clips -- but symmetrized: each leg
    joint's rotation is the average of the right side and the mirrored left
    side, then mirrored back, so real knee/ankle angles survive while the
    dancer's asymmetric stance (~2 cm) does not
  * upper arms vertical, pointing at the ground   (world -Y)
  * forearms horizontal, pointing forward          (world +Z), 90-deg elbow
  * hands continue the forearm (no finger joints to orient a palm with)
  * root leveled (no rotation), centred in X/Z, ankles at the data's height

The right side is computed from the rig geometry; the left side is mirrored
(local D = diag(-1, 1, 1)) so the two sides are exactly symmetric.

Y-up axes for this rig:  up = +Y,  down = -Y,  forward = +Z (toes),  lateral = X.

Output: soma_retargeter/configs/sources/mydata3/init_pose.bvh   (single frame)
Run:    conda run -n soma-retargeter python tools/gen_mydata3_init_pose.py
"""
import os
import numpy as np
from scipy.spatial.transform import Rotation as Rot

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
# The mydata3 skeleton (hierarchy + bone offsets) is taken from the authoritative
# raw dataset BVH, NOT from the output file (which we overwrite).
_SRC = os.path.join(_ROOT, "dataset", "my_data3", "PM_dance_002_han.bvh")
_OUT = os.path.join(_ROOT, "soma_retargeter", "configs", "sources", "mydata3", "init_pose.bvh")

DOWN = np.array([0.0, -1.0, 0.0])     # upper arm points at the ground
FORWARD = np.array([0.0, 0.0, 1.0])   # forearm points forward


def parse(path):
    lines = open(path).read().splitlines()
    m = next(i for i, l in enumerate(lines) if l.strip() == "MOTION")
    hier = lines[:m]
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


def main():
    hier, joints, offsets, ends = parse(_SRC)
    names = [j[0] for j in joints]
    parent = {j[0]: j[1] for j in joints}
    chans = {j[0]: j[2] for j in joints}
    D = np.diag([-1.0, 1.0, 1.0])
    cap_L, cap_root = read_frame0(_SRC, joints, chans)

    # Upper body (root/spine/neck/head/shoulders) starts leveled at identity; the
    # LEGS + FEET are taken from the data's own standing frame so the zero pose
    # grounds exactly like the motion clips (see gen_mydata2_init_pose.py), but
    # symmetrized: average the right rotation with the mirrored left one, then
    # mirror that mean back onto the left, so the captured knee/ankle flexion
    # survives while the stance becomes exactly L/R symmetric.
    L = {n: np.eye(3) for n in names}
    for chain in ("UpLeg", "Leg", "Foot", "ToeBase"):
        m = Rot.mean(Rot.from_matrix(np.stack([
            cap_L["Right" + chain],
            D @ cap_L["Left" + chain] @ D]))).as_matrix()
        L["Right" + chain] = m
        L["Left" + chain] = D @ m @ D

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

    # 3) hand: no finger joints on this rig, so the hand simply continues the
    #    forearm (identity local). The Hand is a position effector; without
    #    fingers there is nothing that depends on palm orientation.
    L[side + "Hand"] = np.eye(3)

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

    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    with open(_OUT, "w") as f:
        f.write("\n".join(hier))
        f.write("\nMOTION\nFrames: 1\nFrame Time: 0.005556\n")
        f.write(" ".join(f"{v:.6g}" for v in vals) + "\n")

    _verify()


def _verify():
    hier, joints, offsets, ends = parse(_OUT)
    names = [j[0] for j in joints]
    chans = {j[0]: j[2] for j in joints}
    lines = open(_OUT).read().splitlines()
    m = next(i for i, l in enumerate(lines) if l.strip() == "MOTION")
    vals = [float(x) for x in lines[m + 3].split()]
    L, pos = {}, 0
    for name, _, _ in joints:
        cs = chans[name]
        o = order(cs)
        ang = [vals[pos + k] for k, ch in enumerate(cs) if "rotation" in ch.lower()]
        L[name] = Rot.from_euler(o, ang, degrees=True).as_matrix() if o else np.eye(3)
        pos += len(cs)
    G, P = fk(joints, offsets, L)
    ys = [P[n][1] for n in names] + [P[pj][1] + off[1] for pj, off in ends]
    print(f"[verify] standing height {max(ys) - min(ys):.2f} cm, feet at y={min(ys):.3f}")
    for s in ("Right", "Left"):
        u = P[s + "ForeArm"] - P[s + "Arm"]; u /= np.linalg.norm(u)
        fdir = P[s + "Hand"] - P[s + "ForeArm"]; fdir /= np.linalg.norm(fdir)
        print(f"  {s:5s} upperarm {np.round(u,2)} forearm {np.round(fdir,2)}")
    print(f"[OK] wrote {_OUT}")


if __name__ == "__main__":
    main()
