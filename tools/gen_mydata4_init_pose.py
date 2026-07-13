"""
gen_mydata4_init_pose.py  --  Build a SOMA-style symmetric "holding a box" zero
pose ON THE MYDATA4 SKELETON, for use as the mydata4 pipeline initialization_pose.

mydata4 (dataset/kongfang_take_002.bvh) is a standard Y-up centimetre BVH with
the same 60-joint naming as mydata (Hips->Spine..Spine3, full finger chains,
ToeBase), but a different rig convention: the rest pose lays EVERY bone along
its local +Z axis (a straight line), so the standing pose is encoded entirely
in rotations, and every joint carries 6 channels (position + ZXY rotation).

Frame 0 of the capture is a clean standing T-pose facing world -Z (toe
directions and the left/right joint placement -- left at -X, right at +X --
both confirm the -Z facing; the 180-deg yaw vs SOMA/mydata2 is registered in
pipelines/utils.py). We start from it and rebuild the SOMA-style stance:

  * root snapped exactly upright (rig +Z -> world +Y), keeping the -Z facing
  * spine / neck / head straightened (identity locals = vertical in this rig)
  * legs aimed straight down; feet keep the REAL frame-0 ankle pitch -- like
    gen_lafan1_init_pose.py, we take the one low-noise signal frame 0 gives
    us (how much lower the toe sits than the ankle), turn it into a pure
    forward-pitch tilt (no roll/yaw), and average both sides so the stance
    stays symmetric, instead of forcing a fabricated flat foot
  * upper arms vertical, pointing at the ground   (world -Y)
  * forearms horizontal, pointing forward          (world -Z), 90-deg elbow
  * hands continue the forearm; fingers keep their captured (straight) locals
  * root centred at x=z=0 and grounded (lowest point at y=0)

The right side is computed from the rig geometry; the left side is mirrored
(local D = diag(-1, 1, 1)) so the two sides are exactly symmetric.

Child position channels are written as the bone OFFSET values (as in the source
data), since the BVH loader prefers position channels over offsets when present.

Output: soma_retargeter/configs/sources/mydata4/init_pose.bvh   (single frame)
Run:    conda run -n soma-retargeter python tools/gen_mydata4_init_pose.py
"""
import os
import numpy as np
from scipy.spatial.transform import Rotation as Rot

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
_SRC = os.path.join(_ROOT, "dataset", "my_data4", "kongfang_take_002.bvh")
_OUT = os.path.join(_ROOT, "soma_retargeter", "configs", "sources", "mydata4", "init_pose.bvh")

DOWN = np.array([0.0, -1.0, 0.0])     # upper arms / legs point at the ground
FORWARD = np.array([0.0, 0.0, -1.0])  # forearms / toes point forward (actor faces -Z)

# Exact upright root: maps rig +Z (rest bone direction) to world +Y, keeping
# world +Z as the facing direction (this is Rx(-90), what the capture's root
# rotation approximates).
ROOT_UPRIGHT = np.array([[1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0],
                         [0.0, -1.0, 0.0]])


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
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    hier, joints, offsets, ends = parse(_SRC)
    names = [j[0] for j in joints]
    parent = {j[0]: j[1] for j in joints}
    chans = {j[0]: j[2] for j in joints}
    D = np.diag([-1.0, 1.0, 1.0])
    cap_L, _ = read_frame0(_SRC, joints, chans)

    # Start from the captured T-pose, then clean it up chain by chain.
    L = {n: cap_L[n].copy() for n in names}
    L[names[0]] = ROOT_UPRIGHT.copy()
    for n in ("Spine", "Spine1", "Spine2", "Spine3", "Neck", "Neck1", "Head"):
        L[n] = np.eye(3)

    def aim(joint, child, target):
        """Swing `joint` so the bone toward `child` points along world `target`."""
        G, P = fk(joints, offsets, L)
        u = G[joint] @ offsets[child]
        L[joint] = np.linalg.inv(G[parent[joint]]) @ (swing(u, target) @ G[joint])

    side = "Right"
    other = "Left"

    # Real frame-0 foot pitch (cf. gen_lafan1_init_pose.py): the captured
    # toe-below-ankle height drop is the only unambiguous, low-noise foot
    # signal in frame 0, so convert it into a pure forward-pitch direction
    # via trig against the toe bone length, averaged over both sides.
    _, P_cap = fk(joints, offsets, cap_L)
    pitches = []
    for s in (side, other):
        drop = P_cap[s + "Foot"][1] - P_cap[s + "ToeBase"][1]
        length = np.linalg.norm(offsets[s + "ToeBase"])
        pitches.append(np.arcsin(np.clip(drop / length, -1.0, 1.0)))
    foot_pitch = float(np.mean(pitches))
    foot_target = FORWARD * np.cos(foot_pitch) + DOWN * np.sin(foot_pitch)

    # Legs: thigh + shin straight down, foot pitched by the real ankle angle;
    # the toe keeps its captured local (frame 0 is a clean standing T-pose).
    for s in (side, other):
        aim(s + "UpLeg", s + "Leg", DOWN)
        aim(s + "Leg", s + "Foot", DOWN)
        aim(s + "Foot", s + "ToeBase", foot_target)

    # Right arm: keep the captured clavicle (lateral in the T-pose), upper arm
    # straight down, forearm forward (90-deg elbow), hand continuing the forearm.
    aim(side + "Arm", side + "ForeArm", DOWN)
    aim(side + "ForeArm", side + "Hand", FORWARD)
    L[side + "Hand"] = np.eye(3)

    # Mirror the whole left arm (clavicle included) from the right side so the
    # two sides are exactly symmetric; fingers keep their captured locals.
    L[other + "Shoulder"] = D @ L[side + "Shoulder"] @ D
    for j in ("Arm", "ForeArm", "Hand"):
        L[other + j] = D @ L[side + j] @ D

    # Ground + centre: lowest point (incl. end sites) at y=0, root at x=z=0.
    G, P = fk(joints, offsets, L)
    ys = [P[n][1] for n in names] + [(P[pj] + G[pj] @ off)[1] for pj, off in ends]
    root_t = np.array([0.0, -min(ys), 0.0])

    vals = []
    for name, _, _ in joints:
        cs = chans[name]
        row = [0.0] * len(cs)
        # Position channels: root gets the grounded translation; children get
        # their bone offsets (the loader prefers position channels to offsets).
        src_t = root_t if name == names[0] else offsets[name]
        for k, ch in enumerate(cs):
            cl = ch.lower()
            if cl == "xposition":
                row[k] = src_t[0]
            elif cl == "yposition":
                row[k] = src_t[1]
            elif cl == "zposition":
                row[k] = src_t[2]
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
        f.write("\nMOTION\nFrames: 1\nFrame Time: 0.00833333\n")
        f.write(" ".join(f"{v:.6g}" for v in vals) + "\n")

    _verify(joints, ends)


def _verify(joints_src, ends_src):
    hier, joints, offsets, ends = parse(_OUT)
    names = [j[0] for j in joints]
    lines = open(_OUT).read().splitlines()
    m = next(i for i, l in enumerate(lines) if l.strip() == "MOTION")
    vals = [float(x) for x in lines[m + 3].split()]
    chans = {j[0]: j[2] for j in joints}
    pos = 0
    L, root_t = {}, np.zeros(3)
    for name, _, _ in joints:
        cs = chans[name]
        o = order(cs)
        ang = [vals[pos + k] for k, ch in enumerate(cs) if "rotation" in ch.lower()]
        L[name] = Rot.from_euler(o, ang, degrees=True).as_matrix() if o else np.eye(3)
        if name == names[0]:
            for k, ch in enumerate(cs):
                cl = ch.lower()
                if cl == "xposition":
                    root_t[0] = vals[pos + k]
                elif cl == "yposition":
                    root_t[1] = vals[pos + k]
                elif cl == "zposition":
                    root_t[2] = vals[pos + k]
        pos += len(cs)
    G, P = fk(joints, offsets, L)
    ys = [P[n][1] + root_t[1] for n in names] + \
         [(P[pj] + G[pj] @ off)[1] + root_t[1] for pj, off in ends]
    print(f"[verify] joints={len(names)}, standing height {max(ys) - min(ys):.2f} cm, "
          f"lowest point at y={min(ys):.3f}")
    spine = P["Head"] - P["Hips"]
    print(f"  spine dir {np.round(spine / np.linalg.norm(spine), 2)}")
    for s in ("Right", "Left"):
        u = P[s + "ForeArm"] - P[s + "Arm"]; u /= np.linalg.norm(u)
        fdir = P[s + "Hand"] - P[s + "ForeArm"]; fdir /= np.linalg.norm(fdir)
        leg = P[s + "Foot"] - P[s + "UpLeg"]; leg /= np.linalg.norm(leg)
        toe = P[s + "ToeBase"] - P[s + "Foot"]; toe /= np.linalg.norm(toe)
        print(f"  {s:5s} upperarm {np.round(u, 2)} forearm {np.round(fdir, 2)} "
              f"leg {np.round(leg, 2)} toe {np.round(toe, 2)}")
    print(f"[OK] wrote {_OUT}")


if __name__ == "__main__":
    main()
