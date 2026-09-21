# LLM-Driven Pick-and-Place on Dobot Magician Lite

A natural-language, vision-guided pick-and-place system for the Dobot Magician Lite. You type a command like *"stack it where you put the last block"*, and the robot detects colored blocks with an overhead camera, reasons about the request with an LLM, validates the generated code in a safety sandbox, and executes the motion.

**Demo video:** https://youtube.com/shorts/ooD1pU13_IY

Built for RAS 545 (Robotics Systems 1) at Arizona State University, Fall 2025.

## How it works

The system is organized into six modules:

| Module | Role |
|---|---|
| `BlockDetector` | Detects red, blue, green, and yellow blocks using HSV segmentation, morphological filtering, and contour-area gating. Assigns stable IDs (`red_1`, `red_2`, ...) in row order. |
| `CoordinateConverter` | Maps camera pixels to robot coordinates using a homography, with a five-point least-squares affine fallback. |
| `SystemState` | JSON-persisted world model: block positions, stacks, gripper status, command history, and the last placement location. Survives restarts. |
| `LLMCodeGenerator` | Sends the current world state and the user's command to an LLM (Llama 3.3 70B via Groq), which returns Python that uses only approved robot functions. |
| `CodeExecutor` | Parses the generated code with Python's `ast` module and rejects any call outside a whitelist of 20 approved functions, and any import other than `time`. |
| `DobotController` | Executes pick and place primitives with safe travel heights and suction control. |

## Safety design

Generated code never reaches the arm unchecked. Every command passes three gates:

1. **Whitelist validation:** the AST check blocks unapproved function calls and imports.
2. **Restricted execution:** code runs with a minimal set of Python built-ins.
3. **Operator confirmation:** the generated code is printed and runs only after the operator approves it.

The robot also moves to a safe height before and after every grasp, and "next to" placements keep at least 15 mm of center-to-center clearance.

## Stacking and memory

Before each placement, the system computes the Z-offset from the current stack height and the measured block height (17.9 mm), so blocks land correctly on multi-tier stacks. Because the world state is saved to disk, the robot understands follow-up references such as "there", "that one", and "the same spot" across commands and sessions.

## Hardware

- Dobot Magician Lite with suction gripper
- Overhead USB camera (640 x 480)

## Setup

```bash
pip install -r requirements.txt
export GROQ_API_KEY="your-key-here"
python pick_and_place.py
```

On startup you choose a calibration mode: homography from `calibration.json`, the built-in five-point affine calibration, or homography from an ArUco `field_calib.yml`.

## Example commands

- `pick up the red block`
- `place it next to the blue block`
- `put another block on top of that`
- `place the red block at the last position`

Special commands: `status`, `history`, `stacks`, `home`, `open camera`, `reset state`, `mode`, `quit`.

## Limitations and next steps

- Performance has been evaluated qualitatively. Latency, placement error, and repeatability have not yet been measured.
- HSV thresholds are tuned for moderate indoor lighting and may need adjustment elsewhere.
- Planned improvements: quantitative accuracy testing, automatic recalibration, and closed-loop verification that each placement succeeded.
