"""
gen_lafan1_init_pose.py  --  Build a symmetric "holding a box" standing zero pose
ON THE LAFAN1 SKELETON, for use as the lafan1 pipeline initialization_pose.

LAFAN1 (dataset/lafan1/dance1_subject2.bvh) is a standard Y-up centimetre BVH
(22 joints, spine = Hips->Spine->Spine1->Spine2->Neck->Head, arms branch off
Spine2 via LeftShoulder/RightShoulder, no fingers). Its rest pose (all
rotations zero) is NOT a T-pose: every main-chain bone offset points along
each joint's own local +X axis regardless of body part (legs/spine/arms all
literally point along local +X at rest), so identity rotations do not give a
standing figure. Frame 0 of the source clip is mid-dance (not a clean stance
either), so -- unlike mydata2 -- we cannot borrow a captured standing frame
for the legs. Instead this whole pose is built analytically: for every main
bone we solve a pure "swing" rotation (bone rest direction -> desired world
direction), the same way mydata2's script does for the arms only.

Standing figure:
  * torso stands upright AND faces world +X (not just upright at some
    arbitrary yaw/backwards-facing twist -- see the Hips comment in main()),
    legs straight down, feet pitched slightly forward-and-down by an angle
    derived from frame 0's real ankle/toe HEIGHTS (not a fabricated flat
    angle, and not the noisier full real rotation either) -- see the Foot
    comment in main()
  * shoulders aimed out to the side (world +-Z) instead of inheriting the
    spine's upright orientation -- otherwise the shoulder joint ends up
    stacked almost directly above the chest instead of beside it, which
    visually pulls both arms in toward the midline
  * upper arms vertical, pointing at the ground   (world -Y)
  * forearms horizontal, pointing forward          (world +X, this rig's
    forward axis -- Z is lateral for this rig, X is forward/back. Which
    side (+Z/-Z) ends up "Left" vs "Right" falls out of the Hips forward
    constraint below rather than being assumed; the shoulder-aiming code
    reads each shoulder's own attachment offset (via the now-fixed Spine2
    orientation) to pick the matching sign, instead of hardcoding it)
  * hands continue the forearm direction (no fingers to orient)
  * root grounded (feet at y=0)

Because every main-chain bone's rest direction is already the same local +X
regardless of side, solving the swing independently per side (no explicit
left/right mirroring) naturally produces an exactly symmetric pose.

Output: soma_retargeter/configs/sources/lafan1/init_pose.bvh   (single frame)
Run:    conda run -n soma-retargeter python tools/gen_lafan1_init_pose.py
"""
import os
import numpy as np
from scipy.spatial.transform import Rotation as Rot

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
_SRC = os.path.join(_ROOT, "dataset", "lafan1", "dance1_subject2.bvh")
_OUT = os.path.join(_ROOT, "soma_retargeter", "configs", "sources", "lafan1", "init_pose.bvh")

DOWN = np.array([0.0, -1.0, 0.0])     # upper arm / leg point at the ground
FORWARD = np.array([1.0, 0.0, 0.0])   # forearm / foot point forward (this rig's axis)
UP = np.array([0.0, 1.0, 0.0])        # spine / neck / head point straight up
LATERAL = np.array([0.0, 0.0, 1.0])   # shoulder points out to the side (+Z = left, -Z = right)


def parse(path):
    lines = open(path).read().splitlines()
    m = next(i for i, l in enumerate(lines) if l.strip() == "MOTION")
    hier = lines[:m]
    joints, offsets, stack = [], {}, []
    cur = pending = None
    ignore = False
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
    return hier, joints, offsets


def order(chans):
    return "".join(c[0].upper() for c in chans if "rotation" in c.lower())


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
    """Rotation mapping unit-ish vector a onto vector b (twist around the
    swing axis is left at whatever align_vectors picks; since both sides
    solve the identical a->b problem the result stays symmetric)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return Rot.align_vectors([b], [a])[0].as_matrix()


def frame(e1, e2):
    """Orthonormal frame [e1, e2_perp, e1 x e2_perp] (Gram-Schmidt on e2)."""
    e1 = e1 / np.linalg.norm(e1)
    e2 = e2 - (e2 @ e1) * e1
    e2 = e2 / np.linalg.norm(e2)
    return np.column_stack([e1, e2, np.cross(e1, e2)])


def load_frame_locals(path, frame_idx=0):
    """Read a BVH's own recorded per-joint local rotations at ``frame_idx``
    (each joint's rotation CHANNELS converted straight to a matrix -- these
    are local-to-parent joint angles, independent of the rest of the chain,
    so they can be read out of one pose and reused verbatim in another)."""
    hier, joints, offsets = parse(path)
    chans = {j[0]: j[2] for j in joints}
    lines = open(path).read().splitlines()
    m = next(i for i, l in enumerate(lines) if l.strip() == "MOTION")
    vals = [float(x) for x in lines[m + 3 + frame_idx].split()]
    L, pos = {}, 0
    for name, _, _ in joints:
        cs = chans[name]
        o = order(cs)
        ang = [vals[pos + k] for k, ch in enumerate(cs) if "rotation" in ch.lower()]
        L[name] = Rot.from_euler(o, ang, degrees=True).as_matrix() if o else np.eye(3)
        pos += len(cs)
    return L


def main():
    hier, joints, offsets = parse(_SRC)
    names = [j[0] for j in joints]
    parent = {j[0]: j[1] for j in joints}
    chans = {j[0]: j[2] for j in joints}

    L = {n: np.eye(3) for n in names}
    # G[name] holds the world orientation to use when placing name's OWN
    # children (i.e. it's the frame that offsets[child] gets transformed by --
    # NOT the frame of the bone leading into `name` itself, which is
    # determined by G[parent(name)] instead).
    G = {}

    # Hips (root): needs TWO constraints, not one. Like every other main-chain
    # bone its offset to its child (Spine) points along local +X regardless of
    # anatomical direction, so that alone must swing to point straight up --
    # but a single-vector swing leaves the twist *around* that up-axis free,
    # and Rot.align_vectors happens to pick a twist where the character's
    # front ends up facing backwards relative to the FORWARD target we aim
    # the feet/forearms at below (verified empirically: this rig's rotation
    # convention has the ROOT's local +Y axis, not +X or +Z, as "forward" --
    # matching what the LAFAN1 paper documents -- and a plain swing(Spine
    # offset, UP) maps that local +Y to a mostly-backward direction). Solve
    # for BOTH constraints at once (offsets["Spine"] -> UP, local +Y ->
    # FORWARD) via two Gram-Schmidt frames, which pins down the twist too.
    # NOTE: this specific choice (+FORWARD, not -FORWARD) is load-bearing --
    # verified empirically to be the one that keeps LeftUpLeg/RightUpLeg
    # retargeting to sane hip-roll ranges (the other sign choice is an
    # equally "internally valid" pose in isolation, but corresponds to a
    # 180-degree-rotated standing orientation that the leg IK map was not
    # calibrated against).
    S = frame(offsets["Spine"], np.array([0.0, 1.0, 0.0]))
    T = frame(UP, FORWARD)
    G["Hips"] = T @ S.T
    L["Hips"] = G["Hips"]

    # Spine1/Spine2/Neck/Head: same "offset points along local +X" issue.
    # IMPORTANT: unlike what the old comment here claimed, the twist about
    # the up-axis is NOT a harmless nuance -- computing each G_target as an
    # independent WORLD-frame swing(rest, UP) lets align_vectors pick an
    # UNRELATED (and empirically often ~180 deg different) twist at each
    # segment, decoupled from the twist Hips was just pinned to. Since
    # "Spine2" and "Hips" are BOTH mapped to the robot's single rigid
    # base_link (ik_map), a twist mismatch between them doesn't cancel out --
    # it gets baked into their respective calibrated joint_offsets and then
    # RE-INTRODUCED at retarget time as a large, roughly-constant spurious
    # disagreement between what Hips and Spine2 each ask of base_link
    # (verified: ~175 deg average on the real clip, vs ~20 deg naturally in
    # the raw mocap -- the IK then has to compromise between two targets
    # ~180 deg apart, which looks like "the upper body is off by ~90 deg").
    # Fix: a plain single-vector swing(rest, local_up) is STILL ambiguous --
    # it picks whatever twist align_vectors' minimal-rotation happens to
    # land on, which can just as easily flip 180 deg as agree with the
    # parent (same failure mode as before, just one level removed). Pin
    # down the twist explicitly instead, exactly like Hips did: require
    # BOTH that `rest` swings to "up" AND that this bone's local +Y (this
    # rig's forward axis, per the Hips comment above) continues to point
    # the same *parent-relative* forward direction the parent itself uses.
    # Both target directions are expressed in the parent's own rotated
    # frame (inv(G_grandparent) @ world_dir), so the twist propagates
    # continuously all the way down from Hips's explicit anchor instead of
    # being re-decided independently at each segment. This keeps Hips and
    # Spine2 in close agreement (~20 deg, matching real mocap) instead of
    # the ~180 deg mismatch an independent swing(rest, UP) per segment used
    # to produce -- important because both map to the robot's single rigid
    # base_link in the ik_map, so any twist disagreement between them
    # doesn't cancel out, it gets IK-averaged into a large, roughly-static
    # spurious "upper body" rotation error on every retargeted frame.
    G_grandparent = G["Hips"]
    spine_chain = [("Spine", "Spine1"), ("Spine1", "Spine2"), ("Spine2", "Neck"), ("Neck", "Head")]
    for par, child in spine_chain:
        rest = offsets[child]
        local_up = np.linalg.inv(G_grandparent) @ UP
        local_forward = np.linalg.inv(G_grandparent) @ FORWARD
        S = frame(rest, np.array([0.0, 1.0, 0.0]))
        T = frame(local_up, local_forward)
        L[par] = T @ S.T
        G[par] = G_grandparent @ L[par]
        G_grandparent = G[par]

    # Real frame-0 foot pitch, borrowed for the Foot joint below. Earlier
    # versions of this script tried transplanting the ankle's FULL 3-DOF
    # rotation from frame 0 (either the raw local channel, or the whole
    # orientation relative to Hips) -- both ended up carrying over noisy
    # roll/yaw artifacts specific to that exact mid-dance instant (leaning,
    # turned-out toes, etc.), producing a foot tilted ~39 deg down AND ~22
    # deg sideways: technically "real data" but a much more extreme, odd-
    # looking stance than the actual clip ever needs as a rest reference.
    # Instead, take just the ONE unambiguous, low-noise signal frame 0
    # actually gives us: how much LOWER the toe sits than the ankle (a
    # simple height/absolute-position fact, not a rotation), and turn that
    # into a pure forward-pitch tilt -- no roll, no yaw -- via basic trig
    # against the known foot-bone length. Left and Right differ slightly on
    # frame 0 (mid-dance isn't symmetric); average them for a properly
    # symmetric reference.
    real_L = load_frame_locals(_SRC, frame_idx=0)
    _, P_real = fk(joints, offsets, real_L)
    pitches = []
    for side in ("Left", "Right"):
        drop = P_real[side + "Foot"][1] - P_real[side + "Toe"][1]
        length = np.linalg.norm(offsets[side + "Toe"])
        pitches.append(np.arcsin(np.clip(drop / length, -1.0, 1.0)))
    foot_pitch = float(np.mean(pitches))
    FOOT_TARGET = np.array([np.cos(foot_pitch), -np.sin(foot_pitch), 0.0])

    for side in ("Left", "Right"):
        # Upper leg: hip -> knee bone points down. Hips' own G is the
        # (now upright, not identity) spine orientation computed above.
        rest = offsets[side + "Leg"]
        G_upleg = swing(rest, DOWN)
        L[side + "UpLeg"] = np.linalg.inv(G["Hips"]) @ G_upleg
        G[side + "UpLeg"] = G_upleg

        # Shin: knee -> ankle bone continues straight down (no knee bend).
        rest = offsets[side + "Foot"]
        G_leg = swing(rest, DOWN)
        L[side + "Leg"] = np.linalg.inv(G_upleg) @ G_leg
        G[side + "Leg"] = G_leg

        # Foot: swing to FOOT_TARGET (computed above) instead of pure
        # FORWARD -- a small, real-data-derived forward pitch instead of
        # either a fabricated "perfectly flat" angle or the noisy full real
        # rotation this used to transplant.
        rest = offsets[side + "Toe"]
        G_foot = swing(rest, FOOT_TARGET)
        L[side + "Foot"] = np.linalg.inv(G_leg) @ G_foot
        G[side + "Foot"] = G_foot
        # Toe stays flat, continuing the foot's own direction (it isn't
        # mapped to anything in the robot's ik_map anyway, so this is purely
        # cosmetic for the skeleton viewer).
        L[side + "Toe"] = np.eye(3)

        # Shoulder: like every other main-chain bone, this rig's rest offset
        # points along local +X regardless of anatomical direction, so it is
        # NOT a small clavicle nub -- it is ~20cm, dominant in X. Leaving its
        # rotation at identity (continuing Spine2's now-upright orientation)
        # swings that +X offset to point straight UP, stacking the shoulder
        # joint almost directly above the chest (only ~6cm of true sideways
        # spread) instead of beside it -- visually pulling both arms in
        # toward the midline. Aim it explicitly out to the side instead --
        # but "out to the side" must agree with whichever world-Z sign this
        # bone's OWN attachment offset (offsets[side+"Shoulder"], transformed
        # by the now-fixed G["Spine2"]) actually lands on, not an assumed
        # Left=+Z convention: Hips's forward-facing twist constraint above
        # determines that sign, and getting it backwards here doesn't show
        # up as an "unreasonable" pose (still upright, still symmetric) --
        # it just points each upper arm in exactly the wrong direction,
        # which downstream only shows up as the whole arm chain looking
        # twisted, not as an obviously-broken number.
        attach_z = (G["Spine2"] @ offsets[side + "Shoulder"])[2]
        lateral = LATERAL if attach_z > 0 else -LATERAL
        rest = offsets[side + "Shoulder"]
        G_shoulder = swing(rest, lateral)
        L[side + "Shoulder"] = np.linalg.inv(G["Spine2"]) @ G_shoulder
        G[side + "Shoulder"] = G_shoulder

        # Upper arm: shoulder -> elbow bone points down. Composed against the
        # shoulder's actual (upright, not identity) world orientation.
        rest = offsets[side + "ForeArm"]
        G_arm = swing(rest, DOWN)
        L[side + "Arm"] = np.linalg.inv(G_shoulder) @ G_arm
        G[side + "Arm"] = G_arm

        # Forearm: elbow -> wrist bone bends 90 deg forward.
        rest = offsets[side + "Hand"]
        G_forearm = swing(rest, FORWARD)
        L[side + "ForeArm"] = np.linalg.inv(G_arm) @ G_forearm
        G[side + "ForeArm"] = G_forearm

        # Hand: no fingers to orient, just continue the forearm.
        L[side + "Hand"] = np.eye(3)

    # Ground the character: root translation lifts the LOWEST foot point to
    # y=0. Used to just be the ankle joint's own height, which was fine back
    # when the foot bone was forced horizontal (ankle height ~= toe height)
    # -- but now the foot is aimed at frame 0's real (steeply forward/down
    # pointing) angle, so the toe sits well below the ankle, and grounding
    # by ankle alone left the toe (and part of the sole) sunk below y=0.
    G, P = fk(joints, offsets, L)
    lowest_y = min(P["LeftFoot"][1], P["RightFoot"][1], P["LeftToe"][1], P["RightToe"][1])
    root_t = np.array([0.0, -lowest_y, 0.0])

    vals = []
    for name, _, _ in joints:
        cs = chans[name]
        row = [0.0] * len(cs)
        if name == names[0]:
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
        f.write("\nMOTION\nFrames: 1\nFrame Time: 0.033333\n")
        f.write(" ".join(f"{v:.6g}" for v in vals) + "\n")

    _verify()


def _verify():
    hier, joints, offsets = parse(_OUT)
    names = [j[0] for j in joints]
    L = load_frame_locals(_OUT, frame_idx=0)
    G, P = fk(joints, offsets, L)
    # fk() always roots the FIRST joint at (0,0,0) -- it never reads the
    # ROOT's own Xposition/Yposition/Zposition channels (those only make
    # sense for the root, which has no parent offset to apply them to), so
    # the grounding root_t written by main() has to be re-added by hand here
    # to see the actual on-screen height instead of a pre-grounding one.
    lines = open(_OUT).read().splitlines()
    m = next(i for i, l in enumerate(lines) if l.strip() == "MOTION")
    root_t = np.array([float(x) for x in lines[m + 3].split()[0:3]])
    for n in P:
        P[n] = P[n] + root_t
    print(f"[verify] standing height {(P['Head'][1] - min(P['LeftToe'][1], P['RightToe'][1])) / 100.0:.2f} m "
          f"(lowest point at y={min(P['LeftFoot'][1], P['RightFoot'][1], P['LeftToe'][1], P['RightToe'][1]):.3f}, "
          f"ankles at y={min(P['LeftFoot'][1], P['RightFoot'][1]):.3f})")
    for s in ("Left", "Right"):
        u = P[s + "ForeArm"] - P[s + "Arm"]; u /= np.linalg.norm(u)
        fdir = P[s + "Hand"] - P[s + "ForeArm"]; fdir /= np.linalg.norm(fdir)
        leg = P[s + "Foot"] - P[s + "Leg"]; leg /= np.linalg.norm(leg)
        foot = P[s + "Toe"] - P[s + "Foot"]; foot /= np.linalg.norm(foot)
        print(f"  {s:5s} upperarm {np.round(u, 2)} forearm {np.round(fdir, 2)} "
              f"shin {np.round(leg, 2)} foot {np.round(foot, 2)}")
    print(f"  shoulder width {abs(P['LeftShoulder'][2] - P['RightShoulder'][2]) / 100.0:.2f} m, "
          f"hip width {abs(P['LeftUpLeg'][2] - P['RightUpLeg'][2]) / 100.0:.2f} m")
    chest_facing = P["Head"] - P["Hips"]
    chest_facing[1] = 0.0
    chest_facing /= np.linalg.norm(chest_facing)
    shoulder_facing = np.cross(UP, P["LeftShoulder"] - P["RightShoulder"])
    shoulder_facing /= np.linalg.norm(shoulder_facing)
    print(f"  chest facing (Hips->Head, horizontal) {np.round(chest_facing, 2)}, "
          f"shoulder-line facing {np.round(shoulder_facing, 2)} (both should be ~FORWARD {FORWARD})")
    print(f"[OK] wrote {_OUT}")


if __name__ == "__main__":
    main()
