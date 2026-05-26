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

Set `retarget_target` in the config JSON, or pick the matching config file on the command line.

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
```

![Interactive viewer](assets/docs/interactive-viewer-screenshot.png)

**Typical workflow**

1. **Scene Options** (top-right): **Load** a `.bvh` → **Retarget** → **Save** CSV.
2. Use **Playback Controls** (bottom) to scrub, change speed, or loop.
3. Toggle mesh / skeleton / joint axes / gizmos under **Visibility**.

**Right Panels**

- **Calibration (Compute Bias)** — match the robot zero pose to SOMA, then compute/write `joint_scales` and `joint_offsets` in the scaler config. Enable *Calibration Mode* to freeze playback and edit joints.
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

- `assets/selected_sonic_g1_config.json`
- `assets/selected_sonic_pm01_config.json`
- `assets/selected_sonic_pnd_config.json`

### CLI calibration (optional)

Equivalent to the in-viewer calibration buttons:

```bash
python tools/calibrate_robot_offsets.py <robot_type> --scales --calc-pos --write
```

Use the GUI calibration panel when tuning a new robot’s scaler config for the first time.

## Project layout

| Path | Role |
|------|------|
| `app/bvh_to_csv_converter.py` | Entry point (GUI + batch) |
| `soma_retargeter/pipelines/` | Retargeting, IK, feet stabilization |
| `soma_retargeter/robotics/` | Human-to-robot scaling |
| `soma_retargeter/configs/` | Per-robot scaler / retargeter JSON |
| `soma_retargeter/renderers/` | Viewer drawing helpers |
| `assets/robots/` | Robot MJCF / meshes |

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
