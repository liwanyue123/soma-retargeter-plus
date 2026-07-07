# SOMA Retargeter
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

![SOMA Retargeter Banner](assets/docs/banner.gif)

Convert [SOMA](https://github.com/NVlabs/SOMA-X) human motion captures into humanoid robot joint animation. Input is BVH motion on the SOMA skeleton; output is robot-playable CSV joint trajectories. Retargeting uses GPU IK via [Newton](https://github.com/newton-physics/newton) and [NVIDIA Warp](https://github.com/NVIDIA/warp).

The pipeline applies human-to-robot scaling, multi-objective IK with joint limits, feet stabilization, and per-DOF clamping.

> **Note:** Active development — APIs and configs may change between releases.

## Supported robots

| `retarget_target` | Config | Notes |
|-------------------|--------|--------|
| `unitree_g1` | `assets/default_bvh_to_csv_converter_config.json` | Unitree G1 (29 DOF) |
| `engineai_pm01` | `assets/pm01_bvh_to_csv_converter_config.json` | EngineAI PM01 (24 DOF) |
| `hightorque_pi_plus` | `assets/pi_plus_bvh_to_csv_converter_config.json` | Hightorque Pi Plus (20 DOF) |
| `pndbotics_adam_lite` | `assets/adam_lite_bvh_to_csv_converter_config.json` | PND Adam Lite (25 DOF) |
| `pndbotics_adam_sp` | `assets/adam_sp_bvh_to_csv_converter_config.json` | PND Adam SP (29 DOF; 3-DOF waist, 7-DOF arms) |

Set `retarget_target` in the config JSON, or pick the matching config file on the command line.

> `pndbotics_adam_sp` ships as a URDF (loaded via Newton's `add_urdf`); all others are MJCF. Loading its `.dae` meshes needs `pycollada` (`pip install pycollada`).

## Requirements

- **Python** 3.12
- **Git LFS** (for meshes and sample motions)
- **OS** Windows (x86-64), Linux (x86-64, aarch64)
- **GPU** NVIDIA Maxwell or newer, driver 545+ (CUDA 12). No local CUDA Toolkit required.

## Installation

<details>
<summary>Setup (conda or uv)</summary>

### conda + pip

```bash
conda create -n soma-retargeter python=3.12 -y
conda activate soma-retargeter
cd soma-retargeter-plus
git lfs pull
pip install --extra-index-url https://pypi.nvidia.com .
```

`warp-lang` is hosted on NVIDIA’s PyPI index — include `--extra-index-url` on install.

**Linux GUI:** install tkinter if file dialogs fail:

```bash
sudo apt-get install python3.12-tk
```

**Windows:** if `imgui-bundle` fails to install, install the [VC++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

### uv

```bash
git lfs pull
uv sync
```

Use `uv run` instead of `python` in the commands below when using uv.

### Verify

```bash
python -c "import newton, warp, soma_retargeter; print('OK')"
```

</details>

## Sample data

Ten sample BVH/CSV pairs live under `assets/motions/` for smoke tests.

For larger SOMA-skeleton datasets, see the [SEED dataset](https://huggingface.co/datasets/bones-studio/seed) ([Bones Studio](https://huggingface.co/bones-studio)). G1 motions in SEED were retargeted with this tool.

## Quick start

### Interactive viewer

```bash
conda activate soma-retargeter
cd soma-retargeter-plus

# Default: Unitree G1
python app/bvh_to_csv_converter.py \
  --config assets/default_bvh_to_csv_converter_config.json \
  --viewer gl
```

Other robots — same command, different config:

```bash
python app/bvh_to_csv_converter.py --config assets/pm01_bvh_to_csv_converter_config.json --viewer gl
python app/bvh_to_csv_converter.py --config assets/pi_plus_bvh_to_csv_converter_config.json --viewer gl
python app/bvh_to_csv_converter.py --config assets/adam_lite_bvh_to_csv_converter_config.json --viewer gl
python app/bvh_to_csv_converter.py --config assets/adam_sp_bvh_to_csv_converter_config.json --viewer gl
```

To retarget your own (non-SOMA) skeleton directly, add `--data <source>` — see [Custom (non-SOMA) data](#custom-non-soma-data).

![Interactive viewer](assets/docs/interactive-viewer-screenshot.png)

**Typical workflow**

1. **Scene Options** (top-right): **Load** a `.bvh` → **Retarget** → **Save** CSV.
2. Use **Playback Controls** (bottom) to scrub, change speed, or loop.
3. Toggle mesh / skeleton / joint axes / gizmos under **Visibility**.

**Right Panels**

- **Calibration (Compute Bias)** — match the robot zero pose to the source zero pose, then compute/write `joint_scales` and `joint_offsets` (see [Custom data](#custom-non-soma-data) for where each is stored per source). Enable *Calibration Mode* to freeze playback and edit joints.
- **Scene Objects** — place reference boxes in the viewport (size in meters, drag gizmo or type position). **Save…** / **Load…** writes a JSON scene file.

### Batch (headless)

Edit `import_folder` and `export_folder` in the config, then:

```bash
python app/bvh_to_csv_converter.py \
  --config assets/default_bvh_to_csv_converter_config.json \
  --viewer null
```

All `.bvh` files under `import_folder` are processed recursively; CSVs are written under `export_folder` with the same folder layout.

Pre-made batch configs for SONIC-style exports:

```bash
# Unitree G1
python app/bvh_to_csv_converter.py \
  --config assets/selected_sonic_g1_config.json \
  --viewer null

# EngineAI PM01
python app/bvh_to_csv_converter.py \
  --config assets/selected_sonic_pm01_config.json \
  --viewer null

# Hightorque Pi Plus
python app/bvh_to_csv_converter.py \
  --config assets/selected_sonic_pi_config.json \
  --viewer null

# PND Adam Lite
python app/bvh_to_csv_converter.py \
  --config assets/selected_sonic_pnd_config.json \
  --viewer null
```

### CLI calibration (optional)

Equivalent to the in-viewer calibration buttons:

```bash
python tools/calibrate_robot_offsets.py <robot_type> --scales --calc-pos --write
```

Use the GUI calibration panel when tuning a new robot’s scaler config for the first time.

## Custom (non-SOMA) data

You can retarget **your own mocap skeleton directly**, without first converting it to
the SOMA skeleton (any such conversion re-bakes rotations onto different bone
proportions and introduces foot sliding/drift, even from clean source data). Select a
custom source with the `--data` flag:

```bash
# Unitree G1
python app/bvh_to_csv_converter.py \
  --config assets/default_bvh_to_csv_converter_config.json \
  --viewer gl \
  --data mydata

# EngineAI PM01
python app/bvh_to_csv_converter.py \
  --config assets/pm01_bvh_to_csv_converter_config.json \
  --viewer gl \
  --data mydata

# Hightorque Pi Plus
python app/bvh_to_csv_converter.py \
  --config assets/pi_plus_bvh_to_csv_converter_config.json \
  --viewer gl \
  --data mydata

# PND Adam Lite
python app/bvh_to_csv_converter.py \
  --config assets/adam_lite_bvh_to_csv_converter_config.json \
  --viewer gl \
  --data mydata

# PND Adam SP
python app/bvh_to_csv_converter.py \
  --config assets/adam_sp_bvh_to_csv_converter_config.json \
  --viewer gl \
  --data mydata
```

`--data <source>` switches the pipeline to a registered non-SOMA source. The source's
own joint names are used end-to-end, the SOMA skin mesh is skipped (the skeleton bones
are drawn instead), and a dedicated retargeter config is loaded. Per-source load
conventions — up-axis/facing, unit scale, horizontal recenter, and yaw offset vs. the
robot — are all registered in `soma_retargeter/pipelines/utils.py` so onboarding a new
source is mostly a matter of filling in tables + JSON configs.

> **Each `--data` source is wired up per robot.** Every `(source, robot)` pair has its own
> entry in `_RETARGETER_CONFIG_FILENAME` plus a matching `ik_map` / scaler / offsets /
> init-pose config. `mydata` ships with starter configs for all five robots above, but
> their `joint_scales` / `joint_offsets` are still neutral placeholders — run the in-app
> **Calibration** panel once per robot to fill in real values (Adam Lite is already
> calibrated; Adam SP's `mydata` configs are copied from Adam Lite and still need a
> calibration pass). To add another robot or a brand-new source, see
> [Adding a new source/robot pair](#custom-non-soma-data) below.

**Config split per source.** Configs live under `configs/<robot>/<source>/` (e.g.
`configs/engineai_pm01/mydata/`), with the robot-level `<robot>_feet_stabilizer_config.json`
kept at `configs/<robot>/`. For custom sources, *tracking/scaling* and *calibration
offsets* live in separate files so they can be regenerated independently:

| File (under `configs/<robot>/<source>/`) | Holds |
|------|-------|
| `<source>_to_<robot>_scaler_config.json` | `joint_scales` (tracking) + `joint_parents` |
| `<source>_to_<robot>_offsets_config.json` | `joint_offsets` (calibration result) |
| `<source>_to_<robot>_retargeter_config.json` | `ik_map` (keyed by your joint names) + references the two files above via `human_robot_scaler_config` and `joint_offsets_config` |

The retargeter config also points `initialization_pose` at a SOMA-style symmetric
"holding-box" zero pose **on your skeleton** (generate one with
`tools/gen_my_init_pose.py`). SOMA sources keep `joint_offsets` inline in the scaler
config and are unaffected.

**Adding a new source / robot pair.** Steps 1–2 are only needed for a brand-new source
(e.g. `xsens`); to point an existing source like `mydata` at another robot (e.g.
`g1` / `pm01` / `pi_plus`), skip to step 3 and just add a new
`_RETARGETER_CONFIG_FILENAME` entry for `(SourceType.MYDATA, "<robot>")`.

1. Register a new source in `soma_retargeter/pipelines/utils.py`: add a `SourceType`, its
   string in `_SOURCE_TYPE_TO_STR`, and one row in each per-source table (facing
   direction, position scale, recenter, yaw offset).
2. Inspect the source once and fill those tables: up-axis (`Mujoco` for Y-up, `Newton`
   for Z-up), extra unit scale, horizontal recenter, and yaw vs. the robot.
3. Add a `_RETARGETER_CONFIG_FILENAME` entry for `(SourceType.<SOURCE>, "<robot>")`
   pointing at a `<source>_to_<robot>_retargeter_config.json`.
4. Generate the `initialization_pose` on your skeleton with `tools/gen_my_init_pose.py`
   (point its `SRC` / `OUT` / joint names at your rig).
5. Author that retargeter config with an `ik_map` keyed by your joint names → this
   robot's body links, plus starter scaler + offsets configs (`joint_scales` = 1.0,
   identity `joint_offsets`).
6. Launch with `--data <source>` against that robot's config, then use the Calibration
   panel to compute and write the real `joint_scales` / `joint_offsets`.

## Project layout

| Path | Role |
|------|------|
| `app/bvh_to_csv_converter.py` | Entry point (GUI + batch) |
| `soma_retargeter/pipelines/` | Retargeting, IK, feet stabilization |
| `soma_retargeter/robotics/` | Human-to-robot scaling |
| `soma_retargeter/configs/<robot>/<source>/` | Retargeter / scaler / offsets JSON for each `<source>_to_<robot>` pair |
| `soma_retargeter/configs/<robot>/` | Robot-level config (e.g. `<robot>_feet_stabilizer_config.json`) shared across sources |
| `soma_retargeter/configs/sources/` | Per-source skeleton assets (SOMA mesh + zero/T-pose, native init poses) |
| `tools/reference_poses/` | Per-robot calibration reference-pose JSON |
| `soma_retargeter/renderers/` | Viewer drawing helpers |
| `assets/robots/` | Robot MJCF / URDF / meshes |

## Related work

Part of the [SOMA](https://github.com/NVlabs/SOMA-X) ecosystem:

- [SOMA Body Model](https://github.com/NVlabs/SOMA-X)
- [GEM-X](https://github.com/NVlabs/GEM-X)
- [Kimodo](https://github.com/nv-tlabs/kimodo)
- [ProtoMotions](https://github.com/NVlabs/ProtoMotions)
- [SONIC](https://nvlabs.github.io/GEAR-SONIC/)

## Acknowledgments

Inspired by [GMR](https://github.com/YanjieZe/GMR) and [PyRoki](https://pyroki-toolkit.github.io/).

## License

[Apache-2.0](LICENSE). Third-party dependencies have their own licenses — review before redistribution.
