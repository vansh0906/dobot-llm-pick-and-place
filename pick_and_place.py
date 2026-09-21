#!/usr/bin/env python3
"""
LLM-Driven Pick-and-Place on Dobot Magician Lite with Persistent Memory.

Natural-language commands are turned into validated robot actions through
perception (HSV block detection), camera-to-robot calibration, a JSON-backed
world state, LLM code generation, and an AST-based safety sandbox.
"""
import os
import sys
import json
import time
import ast
import cv2
import numpy as np
from langchain_groq import ChatGroq  # type: ignore
from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
from datetime import datetime
from pathlib import Path

# Optional YAML support (for ArUco field_calib.yml)
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

# Read the GROQ API key from environment for safety (do NOT hard-code it here)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Hardware ports / indices
DOBOT_PORT = '/dev/ttyACM0'
CAMERA_INDEX = 1

# Camera frame (used for bounds clamping when computing adjacent placements)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# State file for persistent memory (stored next to this script)
STATE_FILE = Path(__file__).with_name("dobot_state.json")

# Dobot "home" position
HOME_POSITION = {'x': 250.0, 'y': 0, 'z': 150.0, 'r': 0.0}

# Calibration / motion constants
PICK_HEIGHT = -54.2
SAFE_HEIGHT = 50.0
PLACE_HEIGHT = -54.2
BLOCK_HEIGHT = 17.9

# In-plane block side (edge length) used for "next to" spacing
BLOCK_SIDE_MM = 15.0  # 1.5 cm
MIN_NEXT_TO_CLEARANCE_MM = BLOCK_SIDE_MM  # never place centers closer than 15 mm

# Calibration points for affine fallback
CALIBRATION_POINTS = [
    {'pixel': (440, 328), 'dobot': (258.13, -78.04)},
    {'pixel': (171, 371), 'dobot': (236.59, 59.59)},
    {'pixel': (285, 282), 'dobot': (281.80, -1.37)},
    {'pixel': (160, 168), 'dobot': (336.37, 58.25)},
    {'pixel': (440, 178), 'dobot': (332.32, -78.12)},
]

CALIBRATION_JSON = Path(__file__).with_name("calibration.json")
FIELD_CALIB_YML = Path(__file__).with_name("field_calib.yml")  # ArUco-style calibration


class CoordinateConverter:
    """
    Converts pixel coordinates from the camera into Dobot XY coordinates.

    Supports:
    - "homography" mode: uses calibration.json or field_calib.yml
    - "affine" mode: uses 5-point affine calibration from CALIBRATION_POINTS
    """

    def __init__(self, mode="homography", calibration_file=CALIBRATION_JSON):
        self.mode = mode
        self.calibration_file = calibration_file
        self.calibration_points = CALIBRATION_POINTS
        self.avg_r = 14.25
        self.H = None  # homography matrix

        if self.mode == "homography":
            if not self.load_homography_from_file():
                print("[Coordinates] Homography load failed, falling back to affine calibration")
                self.mode = "affine"
                self.calculate_transformation()
        else:
            self.calculate_transformation()

        print(f"[Coordinates] ✓ Calibration ready (mode={self.mode})")

    def mm_to_pixels(self, dx_mm=None, dy_mm=None, axis='x'):
        """
        Roughly convert a desired offset in millimeters to pixels using the current calibration.

        Only one of dx_mm or dy_mm needs to be provided.
        axis:
          'x' -> treat movement as horizontal in image space
          'y' -> treat movement as vertical in image space
        """
        if axis not in ('x', 'y'):
            axis = 'x'

        if dx_mm is not None:
            mm = float(abs(dx_mm))
        elif dy_mm is not None:
            mm = float(abs(dy_mm))
        else:
            raise ValueError("mm_to_pixels: you must provide dx_mm or dy_mm")

        if mm == 0.0:
            return 0.0

        # Sample the mapping around the center of the camera frame to estimate mm/pixel
        cx = FRAME_WIDTH // 2
        cy = FRAME_HEIGHT // 2
        step_px = 10  # pixel step for a finite-difference estimate

        if axis == 'x':
            x0, y0 = self.pixel_to_dobot(cx, cy)
            x1, y1 = self.pixel_to_dobot(cx + step_px, cy)
        else:
            x0, y0 = self.pixel_to_dobot(cx, cy)
            x1, y1 = self.pixel_to_dobot(cx, cy + step_px)

        dist_mm = float(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
        if dist_mm < 1e-6:
            # Fallback guess: 1 pixel ≈ 1 mm if calibration appears degenerate
            mm_per_pixel = 1.0
        else:
            mm_per_pixel = dist_mm / step_px

        pixels = mm / mm_per_pixel
        return pixels

    def load_homography_from_file(self):
        """
        Load a 3x3 homography matrix from the calibration file.

        Supported formats:
        1) Legacy calibration.json
           {"homography_calibration": {"homography_matrix": [[...3x3...]]}}
        2) ArUco field_calib.yml / .json
           {"homography": [[...3x3...]], "homography_inv": [[...3x3...]], ...}
        """
        try:
            path_str = str(self.calibration_file)
            data = None

            # Prefer YAML for .yml/.yaml if PyYAML is available, otherwise try JSON
            if path_str.lower().endswith((".yml", ".yaml")) and yaml is not None:
                with open(self.calibration_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
            else:
                # Try JSON first
                try:
                    with open(self.calibration_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    # Fallback to YAML if JSON parse fails
                    if yaml is not None:
                        with open(self.calibration_file, "r", encoding="utf-8") as f:
                            data = yaml.safe_load(f)
                    else:
                        raise

            if not isinstance(data, dict):
                raise ValueError("Calibration file must contain a JSON/YAML object at the top level")

            # Guess which format we are using
            if "homography_calibration" in data:
                # Old calibration.json format
                H_list = data["homography_calibration"]["homography_matrix"]
            elif "homography" in data:
                # ArUco-style file
                H_list = data["homography"]
            else:
                raise KeyError(
                    "Calibration file missing homography data. Expected either "
                    "'homography_calibration'->'homography_matrix' or 'homography'."
                )

            H = np.array(H_list, dtype=float)
            if H.shape != (3, 3):
                raise ValueError(f"Homography matrix must be 3x3, got {H.shape}")

            self.H = H
            print(f"[Coordinates] ✓ Homography loaded from {self.calibration_file}")
            return True

        except Exception as e:
            print(f"[Coordinates] ✗ Failed to load homography: {e}")
            self.H = None
            return False

    def calculate_transformation(self):
        """Compute affine transformation coefficients from CALIBRATION_POINTS."""
        pixel_coords = np.array([
            [p['pixel'][0], p['pixel'][1], 1] for p in self.calibration_points
        ])
        dobot_x = np.array([p['dobot'][0] for p in self.calibration_points])
        dobot_y = np.array([p['dobot'][1] for p in self.calibration_points])

        coeffs_x, _, _, _ = np.linalg.lstsq(pixel_coords, dobot_x, rcond=None)
        self.a11, self.a12, self.b1 = coeffs_x

        coeffs_y, _, _, _ = np.linalg.lstsq(pixel_coords, dobot_y, rcond=None)
        self.a21, self.a22, self.b2 = coeffs_y

        print("[Coordinates] Affine transformation coefficients calculated")

    def set_mode(self, mode):
        """Switch calibration mode at runtime if needed."""
        if mode == self.mode:
            return
        if mode not in ("homography", "affine"):
            print(f"[Coordinates] Unknown mode '{mode}', keeping '{self.mode}'")
            return

        if mode == "homography":
            if self.load_homography_from_file():
                self.mode = "homography"
            else:
                print("[Coordinates] Staying in affine mode due to homography load failure")
        else:
            self.mode = "affine"
            self.calculate_transformation()

        print(f"[Coordinates] ✓ Switched calibration mode -> {self.mode}")

    def pixel_to_dobot(self, pixel_x, pixel_y):
        """Convert a camera pixel coordinate (u, v) to Dobot XY (in mm)."""
        if self.mode == "homography" and self.H is not None:
            v = np.array([pixel_x, pixel_y, 1.0])
            X = self.H @ v
            if abs(X[2]) < 1e-9:
                raise ValueError("Homography produced invalid scale (W ≈ 0)")
            dobot_x = X[0] / X[2]
            dobot_y = X[1] / X[2]
        else:
            # Affine fallback
            dobot_x = self.a11 * pixel_x + self.a12 * pixel_y + self.b1
            dobot_y = self.a21 * pixel_x + self.a22 * pixel_y + self.b2

        print(f"[Coord] Pixel ({pixel_x}, {pixel_y}) -> Dobot ({dobot_x:.1f}, {dobot_y:.1f}) [{self.mode}]")
        return dobot_x, dobot_y


class SystemState:
    """
    Tracks the full "world state" for the robot:
    - Which blocks exist, where they are, and what color they are
    - Which blocks are stacked on top of which
    - Whether the gripper is holding something
    - Recent commands and where the last placement happened

    This state is persisted to disk so the robot can "remember" across runs.
    """

    def __init__(self, state_file=STATE_FILE, coord_converter=None):
        self.state_file = state_file
        self.gripper_status = "EMPTY"
        self.holding_block = None
        self.blocks = {}                # block_id -> metadata
        self.conversation_history = []
        self.stack_map = {}             # loc_key -> [block_ids bottom->top]
        self.stack_meta = {}            # loc_key -> {'x', 'y', 'height', 'updated_at'}
        self.last_place_location = None
        self.last_manipulated_block_id = None  # used to resolve "that"/"it"
        self.min_adjacent_clearance_mm = MIN_NEXT_TO_CLEARANCE_MM

        # Share the coordinate converter with the Dobot controller when possible
        if coord_converter is None:
            self.coord_converter = CoordinateConverter()
        else:
            self.coord_converter = coord_converter

    def _location_key(self, pixel_x, pixel_y, tolerance=30):
        """
        Bucket blocks into approximate locations so all blocks in the same small
        area share a stack key.
        """
        x_bucket = round(pixel_x / tolerance) * tolerance
        y_bucket = round(pixel_y / tolerance) * tolerance
        return f"{x_bucket},{y_bucket}"

    def _update_stack_meta_for_key(self, loc_key):
        """Recompute centroid and height for a given stack location key."""
        if loc_key not in self.stack_map or len(self.stack_map[loc_key]) == 0:
            if loc_key in self.stack_meta:
                del self.stack_meta[loc_key]
            return

        bids = self.stack_map[loc_key]
        xs = [self.blocks[bid]['pixel_x'] for bid in bids if bid in self.blocks]
        ys = [self.blocks[bid]['pixel_y'] for bid in bids if bid in self.blocks]
        if not xs or not ys:
            return

        cx = int(round(sum(xs) / len(xs)))
        cy = int(round(sum(ys) / len(ys)))
        self.stack_meta[loc_key] = {
            'x': cx,
            'y': cy,
            'height': len(bids),
            'updated_at': datetime.now().isoformat()
        }

    def initialize_blocks(self, detected_blocks):
        """
        Initialize block state from fresh camera detection.
        Existing state is discarded and rebuilt.
        """
        self.blocks = {}
        self.stack_map = {}
        self.stack_meta = {}

        for block in detected_blocks:
            block_id = block['global_id']
            self.blocks[block_id] = {
                'color': block['color'],
                'id': block['id'],
                'pixel_x': block['pixel_x'],
                'pixel_y': block['pixel_y'],
                'original_x': block['pixel_x'],
                'original_y': block['pixel_y'],
                'current_position': {'x': block['pixel_x'], 'y': block['pixel_y'], 'z': 0},
                'stack_level': 0,  # 0 = on table
                'blocks_below': []
            }

            loc_key = self._location_key(block['pixel_x'], block['pixel_y'])
            if loc_key not in self.stack_map:
                self.stack_map[loc_key] = []
            self.stack_map[loc_key].append(block_id)

        # Even if only 1 block at a location, we still keep metadata
        for loc_key in list(self.stack_map.keys()):
            self._update_stack_meta_for_key(loc_key)

        print(f"[State] Initialized {len(self.blocks)} blocks")
        self.save_state()

    def update_gripper(self, block_id=None):
        """
        Update gripper status:
        - If block_id is given -> gripper is now holding that block.
        - If block_id is None -> gripper is empty.
        """
        if block_id is None:
            self.gripper_status = "EMPTY"
        else:
            self.gripper_status = "HOLDING"
            self.holding_block = block_id
            self.last_manipulated_block_id = block_id

        if block_id is None:
            self.holding_block = None

        print(f"[State] Gripper: {self.gripper_status} {block_id or ''}")
        self.save_state()

    def update_block_position(self, block_id, pixel_x, pixel_y, z=0, blocks_below=None):
        """
        Update block position and stack information after a move.

        - Updates pixel position
        - Computes stack level from z offset
        - Updates which blocks are underneath
        - Maintains stack_map and stack_meta
        - Records last placement location and last manipulated block
        """
        if block_id in self.blocks:
            old_loc_key = self._location_key(
                self.blocks[block_id]['pixel_x'],
                self.blocks[block_id]['pixel_y']
            )

            # Update position
            self.blocks[block_id]['pixel_x'] = pixel_x
            self.blocks[block_id]['pixel_y'] = pixel_y
            self.blocks[block_id]['current_position'] = {'x': pixel_x, 'y': pixel_y, 'z': z}

            # Convert z to stack level (0 = table)
            stack_level = round(z / BLOCK_HEIGHT) if z > 0 else 0
            self.blocks[block_id]['stack_level'] = stack_level

            # Update blocks_below
            self.blocks[block_id]['blocks_below'] = (blocks_below or [])

            # Move block between stack locations
            new_loc_key = self._location_key(pixel_x, pixel_y)

            if old_loc_key in self.stack_map and block_id in self.stack_map[old_loc_key]:
                self.stack_map[old_loc_key].remove(block_id)
                if not self.stack_map[old_loc_key]:
                    del self.stack_map[old_loc_key]
                    if old_loc_key in self.stack_meta:
                        del self.stack_meta[old_loc_key]

            if new_loc_key not in self.stack_map:
                self.stack_map[new_loc_key] = []
            if block_id not in self.stack_map[new_loc_key]:
                self.stack_map[new_loc_key].append(block_id)

            # Sort bottom->top by stack level
            self.stack_map[new_loc_key].sort(
                key=lambda bid: self.blocks[bid]['stack_level']
            )

            self._update_stack_meta_for_key(new_loc_key)

            # Record where we just placed something
            self.last_place_location = {'x': pixel_x, 'y': pixel_y, 'z': z}
            self.last_manipulated_block_id = block_id

            print(f"[State] {block_id} moved to ({pixel_x}, {pixel_y}, z={z}, level={stack_level})")
            self.save_state()

    def get_stack_at_location(self, pixel_x, pixel_y):
        """Return list of block_ids (bottom->top) for the stack at a given pixel location."""
        loc_key = self._location_key(pixel_x, pixel_y)
        return self.stack_map.get(loc_key, [])

    def get_stack_height(self, pixel_x, pixel_y):
        """Return how many blocks are in the stack at a given pixel location."""
        stack = self.get_stack_at_location(pixel_x, pixel_y)
        return len(stack)

    def get_block(self, block_id):
        """Return metadata for a given block_id, or None if missing."""
        return self.blocks.get(block_id)

    def is_gripper_empty(self):
        return self.gripper_status == "EMPTY"

    def get_holding_block_id(self):
        return self.holding_block

    def get_last_manipulated_block_id(self):
        """Used when the user says 'that one' or 'it' referring to the previous block."""
        return self.last_manipulated_block_id

    def _distance_to_robot_home_mm(self, pixel_x, pixel_y):
        """Compute Euclidean distance from HOME_POSITION in Dobot XY space."""
        dx, dy = self.coord_converter.pixel_to_dobot(pixel_x, pixel_y)
        return float(np.hypot(dx - HOME_POSITION['x'], dy - HOME_POSITION['y']))

    def _nearest_or_farthest_block(self, block_ids, mode='nearest'):
        """Internal helper to get nearest or farthest block from HOME_POSITION."""
        if not block_ids:
            return None
        scored = []
        for bid in block_ids:
            b = self.blocks.get(bid)
            if not b:
                continue
            d = self._distance_to_robot_home_mm(b['pixel_x'], b['pixel_y'])
            scored.append((d, bid))
        if not scored:
            return None
        scored.sort(key=lambda t: t[0])
        return scored[0][1] if mode == 'nearest' else scored[-1][1]

    def get_nearest_block_id_by_color(self, color):
        ids = [bid for bid, b in self.blocks.items() if b['color'].lower() == color.lower()]
        return self._nearest_or_farthest_block(ids, mode='nearest')

    def get_farthest_block_id_by_color(self, color):
        ids = [bid for bid, b in self.blocks.items() if b['color'].lower() == color.lower()]
        return self._nearest_or_farthest_block(ids, mode='farthest')

    def get_nearest_block_id(self):
        return self._nearest_or_farthest_block(list(self.blocks.keys()), mode='nearest')

    def get_farthest_block_id(self):
        return self._nearest_or_farthest_block(list(self.blocks.keys()), mode='farthest')

    def get_stack_locations(self, min_height=2):
        """
        Return a list of stack descriptors for stacks with height >= min_height:
          [{'loc_key': str, 'x': int, 'y': int, 'height': int}, ...]
        Ordered by tallest first.
        """
        out = []
        for loc_key, meta in self.stack_meta.items():
            h = meta.get('height', 0)
            if h >= min_height:
                out.append({'loc_key': loc_key, 'x': meta['x'], 'y': meta['y'], 'height': h})
        out.sort(key=lambda d: d['height'], reverse=True)
        return out

    def get_tallest_stack_xy(self, min_height=2):
        """Return (x, y, height) for the tallest stack (height>=min_height), or None if none."""
        stacks = self.get_stack_locations(min_height=min_height)
        if not stacks:
            return None
        s = stacks[0]
        return s['x'], s['y'], s['height']

    def get_nearest_stack_xy(self, min_height=2):
        """Return (x, y, height) for the nearest stack to the robot, or None if none."""
        stacks = self.get_stack_locations(min_height=min_height)
        if not stacks:
            return None
        best = None
        bestd = None
        for s in stacks:
            d = self._distance_to_robot_home_mm(s['x'], s['y'])
            if best is None or d < bestd:
                best = s
                bestd = d
        return best['x'], best['y'], best['height']

    def compute_adjacent_position(self, ref_x, ref_y, direction='right', clearance_mm=None):
        """
        Compute a pixel position adjacent to (ref_x, ref_y) with at least clearance_mm
        center-to-center spacing.

        Directions:
          - 'right' -> +x
          - 'left'  -> -x
          - 'front' -> -y (towards top of image)
          - 'back'  -> +y (towards bottom of image)

        Coordinates are clamped to camera frame bounds.
        """
        # Base clearance (with extra safety margin)
        base_clearance = self.min_adjacent_clearance_mm if clearance_mm is None else float(clearance_mm)
        clr = float(base_clearance + 3.0)

        dx_px = dy_px = 0.0
        if direction == 'right':
            dx_px = self.coord_converter.mm_to_pixels(dx_mm=clr, axis='x')
        elif direction == 'left':
            dx_px = -self.coord_converter.mm_to_pixels(dx_mm=clr, axis='x')
        elif direction == 'front':
            dy_px = -self.coord_converter.mm_to_pixels(dy_mm=clr, axis='y')
        elif direction == 'back':
            dy_px = self.coord_converter.mm_to_pixels(dy_mm=clr, axis='y')
        else:
            # default fallback: to the right
            dx_px = self.coord_converter.mm_to_pixels(dx_mm=clr, axis='x')

        nx = int(np.clip(ref_x + dx_px, 0, FRAME_WIDTH - 1))
        ny = int(np.clip(ref_y + dy_px, 0, FRAME_HEIGHT - 1))
        print(f"[State] Adjacent ({direction}, ≥{clr}mm): ({ref_x},{ref_y}) -> ({nx},{ny}) "
              f"[dx_px={dx_px}, dy_px={dy_px}]")
        return nx, ny

    def compute_adjacent_to_block(self, block_id, direction='right', clearance_mm=None):
        """Convenience helper: adjacent position next to a specific block id."""
        b = self.blocks.get(block_id)
        if not b:
            return None
        return self.compute_adjacent_position(b['pixel_x'], b['pixel_y'],
                                              direction=direction, clearance_mm=clearance_mm)

    def compute_adjacent_to_stack(self, direction='right', strategy='tallest', clearance_mm=None):
        """
        Convenience helper: adjacent position next to a chosen stack (height >= 2).

        strategy:
          - 'tallest' (default)
          - 'nearest'
        """
        if strategy == 'nearest':
            sel = self.get_nearest_stack_xy(min_height=2)
        else:
            sel = self.get_tallest_stack_xy(min_height=2)
        if not sel:
            return None
        sx, sy, _ = sel
        return self.compute_adjacent_position(sx, sy, direction=direction, clearance_mm=clearance_mm)

    def add_to_history(self, prompt, action, details=None):
        """Append a user command and its outcome to the conversation history."""
        entry = {
            'prompt': prompt,
            'action': action,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        self.conversation_history.append(entry)
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        self.save_state()

    def get_state_summary(self):
        """Generate a human-readable summary of the system state for the LLM."""
        summary = f"""CURRENT SYSTEM STATE:
- Gripper Status: {self.gripper_status}
- Holding Block: {self.holding_block or 'None'}
- Last Manipulated Block: {self.last_manipulated_block_id or 'None'}

AVAILABLE BLOCKS (with stacking info):
"""
        for block_id, block in self.blocks.items():
            stack_info = ""
            if block['stack_level'] > 0:
                stack_info = (f" [STACKED - Level {block['stack_level']}, "
                              f"on: {', '.join(block['blocks_below'])}]")
            summary += (f"  - {block_id}: {block['color']} at pixel "
                        f"({block['pixel_x']}, {block['pixel_y']}){stack_info}\n")

        # Stacks
        if self.stack_map:
            summary += "\nDETECTED STACKS:\n"
            for loc_key, stack_blocks in self.stack_map.items():
                if len(stack_blocks) > 1:
                    meta = self.stack_meta.get(loc_key, {})
                    cx = meta.get('x', '?')
                    cy = meta.get('y', '?')
                    summary += (f"  - Location {loc_key} @ ({cx},{cy}): "
                                f"{' -> '.join(stack_blocks)} ({len(stack_blocks)} high)\n")

        # Last placement
        if self.last_place_location:
            summary += (f"\nLAST PLACE LOCATION: pixel ({self.last_place_location['x']}, "
                        f"{self.last_place_location['y']}, z={self.last_place_location['z']})\n")

        # Recent actions
        if len(self.conversation_history) > 0:
            summary += f"\nRECENT ACTIONS (last {min(5, len(self.conversation_history))}):\n"
            for entry in self.conversation_history[-5:]:
                details_str = ""
                if entry.get('details'):
                    details_str = f" | {entry['details']}"
                summary += f"  - '{entry['prompt']}' -> {entry['action']}{details_str}\n"

        return summary

    def save_state(self):
        """Persist state to JSON on disk."""
        try:
            state_data = {
                'gripper_status': self.gripper_status,
                'holding_block': self.holding_block,
                'blocks': self.blocks,
                'conversation_history': self.conversation_history,
                'stack_map': self.stack_map,
                'stack_meta': self.stack_meta,
                'last_place_location': self.last_place_location,
                'last_manipulated_block_id': self.last_manipulated_block_id,
                'timestamp': datetime.now().isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            print(f"[State] ✓ Saved to {self.state_file}")
        except Exception as e:
            print(f"[State] Save failed: {e}")

    def load_state(self):
        """Load state from JSON on disk, if it exists."""
        try:
            if not self.state_file.exists():
                print(f"[State] No saved state found at {self.state_file}")
                return False

            with open(self.state_file, 'r') as f:
                state_data = json.load(f)

            self.gripper_status = state_data.get('gripper_status', 'EMPTY')
            self.holding_block = state_data.get('holding_block')
            self.blocks = state_data.get('blocks', {})
            self.conversation_history = state_data.get('conversation_history', [])
            self.stack_map = state_data.get('stack_map', {})
            self.stack_meta = state_data.get('stack_meta', {})
            self.last_place_location = state_data.get('last_place_location')
            self.last_manipulated_block_id = state_data.get('last_manipulated_block_id')

            print(f"[State] ✓ Loaded state from {self.state_file}")
            print(f"[State] Restored: {len(self.blocks)} blocks, "
                  f"{len(self.conversation_history)} history entries")
            return True
        except Exception as e:
            print(f"[State] Load failed: {e}")
            return False


class DobotController:
    """Wraps low-level Dobot Magician Lite control in a friendly API."""

    def __init__(self, calibration_mode="homography", calibration_file=CALIBRATION_JSON):
        self.device = None
        self.coord_converter = CoordinateConverter(
            mode=calibration_mode,
            calibration_file=calibration_file
        )

    def connect(self, port=DOBOT_PORT):
        """Connect to the Dobot arm."""
        try:
            from pydobot import Dobot  # type: ignore
            print(f"\n[Dobot] Connecting to {port}...")
            self.device = Dobot(port=port)
            time.sleep(1)
            if hasattr(self.device, 'set_speed'):
                self.device.set_speed(velocity=150, acceleration=150)
            print("[Dobot] ✓ Connected successfully!")
            return True
        except Exception as e:
            print(f"[Dobot] ✗ ERROR: {e}")
            return False

    def move_to(self, x, y, z, r=0.0, wait=True):
        """Move Dobot to the specified coordinate."""
        if self.device:
            self.device.move_to(x, y, z, r, wait=wait)
            if wait:
                time.sleep(0.2)

    def go_home(self):
        """Return the robot arm to its home position."""
        print("[Dobot] Moving to home position...")
        self.move_to(HOME_POSITION['x'], HOME_POSITION['y'], HOME_POSITION['z'], HOME_POSITION['r'])
        print("[Dobot] ✓ At home position")

    def suction_on(self):
        """Turn suction cup on."""
        if self.device:
            self.device.suck(True)
            print("[Dobot] Suction: ON")
            time.sleep(0.8)

    def suction_off(self):
        """Turn suction cup off."""
        if self.device:
            self.device.suck(False)
            print("[Dobot] Suction: OFF")
            time.sleep(0.5)

    def pick_block(self, pixel_x, pixel_y):
        """Pick up a block located at the given camera pixel."""
        print(f"\n[Dobot] Picking block at pixel ({pixel_x}, {pixel_y})...")
        dobot_x, dobot_y = self.coord_converter.pixel_to_dobot(pixel_x, pixel_y)
        self.move_to(dobot_x, dobot_y, SAFE_HEIGHT)
        self.move_to(dobot_x, dobot_y, PICK_HEIGHT)
        self.suction_on()
        self.move_to(dobot_x, dobot_y, SAFE_HEIGHT)
        print("[Dobot] ✓ Block picked")

    def place_block(self, pixel_x, pixel_y, z_offset=0):
        """Place a block at the given camera pixel and z_offset (for stacking)."""
        print(f"\n[Dobot] Placing block at pixel ({pixel_x}, {pixel_y}), z_offset={z_offset}mm...")
        dobot_x, dobot_y = self.coord_converter.pixel_to_dobot(pixel_x, pixel_y)
        place_z = PLACE_HEIGHT + z_offset
        print(f"[Dobot] Target Z height: {place_z:.1f}mm")
        self.move_to(dobot_x, dobot_y, SAFE_HEIGHT)
        self.move_to(dobot_x, dobot_y, place_z)
        self.suction_off()
        self.move_to(dobot_x, dobot_y, SAFE_HEIGHT)
        print("[Dobot] ✓ Block placed")

    def close(self):
        """Close the Dobot connection cleanly."""
        if self.device:
            try:
                self.device.close()
                print("[Dobot] Connection closed")
            except Exception:
                pass


class BlockDetector:
    """Handles camera input and detects colored blocks in the field of view."""

    VIS_COLORS = {
        'red': (0, 0, 255),
        'blue': (255, 0, 0),
        'green': (0, 255, 0),
        'yellow': (0, 255, 255)
    }

    def __init__(self, camera_index=CAMERA_INDEX):
        self.camera_index = camera_index
        self.camera = None

        # HSV color ranges for simple color segmentation
        self.color_ranges = {
            'red': [
                (np.array([0, 100, 100]), np.array([10, 255, 255])),
                (np.array([160, 100, 100]), np.array([180, 255, 255]))
            ],
            'blue': [(np.array([103, 140, 60]), np.array([130, 255, 255]))],
            'green': [(np.array([40, 100, 100]), np.array([80, 255, 255]))],
            'yellow': [(np.array([20, 100, 100]), np.array([35, 255, 255]))]
        }

    def detect_color_blocks(self, frame, color_name):
        """Detect blocks of a single color in a frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = None
        for lower, upper in self.color_ranges[color_name]:
            if mask is None:
                mask = cv2.inRange(hsv, lower, upper)
            else:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 200 < area < 10000:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    detected.append({'center': (cx, cy), 'contour': contour, 'area': area})
        return detected

    def _detect_and_label(self, frame, display):
        """Detect all colors in a frame, draw labels on display, and return labeled blocks."""
        all_detected = []
        for color_name in ['red', 'blue', 'green', 'yellow']:
            for block in self.detect_color_blocks(frame, color_name):
                all_detected.append({'color': color_name, 'center': block['center'],
                                     'contour': block['contour']})

        # Sort by rough grid order for stable labeling
        all_detected.sort(key=lambda b: (b['center'][1] // 50, b['center'][0]))

        counters = {'red': 1, 'blue': 1, 'green': 1, 'yellow': 1}
        labeled = []
        for block in all_detected:
            cx, cy = block['center']
            color_name = block['color']
            color_bgr = self.VIS_COLORS[color_name]
            color_id = counters[color_name]
            counters[color_name] += 1

            cv2.drawContours(display, [block['contour']], -1, color_bgr, 2)
            cv2.circle(display, (cx, cy), 5, color_bgr, -1)
            label = f"{color_name}_{color_id}"
            cv2.putText(display, label, (cx - 30, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display, f"({cx},{cy})", (cx - 30, cy + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            labeled.append({
                'global_id': label,
                'color': color_name,
                'id': color_id,
                'pixel_x': cx,
                'pixel_y': cy
            })
        return labeled

    def _open_camera(self):
        camera = cv2.VideoCapture(self.camera_index)
        if not camera.isOpened():
            return None
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        return camera

    def detect_all_blocks_live(self):
        """
        Open a live camera window so the user can press SPACE to capture block positions.
        Returns a list of detected blocks with global_ids and pixel coordinates.
        """
        print(f"\n[Camera] Opening camera {self.camera_index}...")
        try:
            self.camera = self._open_camera()
            if self.camera is None:
                print("[Camera] ✗ ERROR: Failed to open camera")
                return None
            print("[Camera] ✓ Camera opened successfully")
        except Exception as e:
            print(f"[Camera] ✗ ERROR: {e}")
            return None

        print("[Camera] Press SPACE to capture | Q to quit")
        detected_blocks = []

        while True:
            ret, frame = self.camera.read()
            if not ret:
                break

            display = frame.copy()
            temp_blocks = self._detect_and_label(frame, display)
            cv2.putText(display, f"Blocks: {len(temp_blocks)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Block Detection", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                if len(temp_blocks) > 0:
                    detected_blocks = temp_blocks
                    print(f"\n[Camera] ✓ Captured {len(detected_blocks)} blocks")
                    break
            elif key == ord('q'):
                print("[Camera] Detection cancelled")
                detected_blocks = None
                break

        self.camera.release()
        cv2.destroyAllWindows()
        return detected_blocks

    def auto_capture_blocks(self, display_duration=2.0):
        """
        Show a short live feed and automatically capture the latest block positions
        after `display_duration` seconds.
        """
        print("\n[Camera] Auto-capturing block positions...")
        try:
            camera = self._open_camera()
            if camera is None:
                return None

            start_time = time.time()
            latest_blocks = []

            while True:
                ret, frame = camera.read()
                if not ret:
                    break

                display = frame.copy()
                latest_blocks = self._detect_and_label(frame, display)

                elapsed = time.time() - start_time
                cv2.putText(display, f"AUTO-CAPTURE - Blocks: {len(latest_blocks)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Auto Block Capture", display)
                cv2.waitKey(30)

                if elapsed >= display_duration:
                    break

            camera.release()
            cv2.destroyAllWindows()
            time.sleep(0.2)
            return latest_blocks

        except Exception as e:
            print(f"[Camera] ✗ ERROR: {e}")
            return None


class LLMCodeGenerator:
    """Uses an LLM to turn natural language instructions into executable Python code."""

    def __init__(self, api_key):
        # Even if key is empty, we instantiate; user will see errors at runtime if needed
        self.llm = ChatGroq(
            temperature=0.1,
            model_name="llama-3.3-70b-versatile",
            api_key=api_key or ""
        )

        self.system_prompt_template = """You are a contextually-aware Python code generator for a Dobot robotic arm.

{state_summary}

ALLOWED FUNCTIONS:
- dobot.pick_block(pixel_x, pixel_y)
- dobot.place_block(pixel_x, pixel_y, z_offset=0)
- system_state.update_gripper(block_id or None)
- system_state.update_block_position(block_id, pixel_x, pixel_y, z, blocks_below=[...])
- system_state.get_block(block_id)
- system_state.get_stack_at_location(pixel_x, pixel_y)
- system_state.get_stack_height(pixel_x, pixel_y)
- system_state.is_gripper_empty()
- system_state.get_holding_block_id()
- system_state.get_last_manipulated_block_id()
- system_state.get_nearest_block_id_by_color(color_name)
- system_state.get_farthest_block_id_by_color(color_name)
- system_state.get_nearest_block_id()
- system_state.get_farthest_block_id()
- system_state.get_tallest_stack_xy()   # returns (x, y, height) for tallest stack (height>=2)
- system_state.get_nearest_stack_xy()   # returns (x, y, height) for nearest stack (height>=2)
- system_state.compute_adjacent_position(x, y, direction='right'|'left'|'front'|'back', clearance_mm=15.0)
- system_state.compute_adjacent_to_block(block_id, direction='right'|'left'|'front'|'back', clearance_mm=15.0)
- system_state.compute_adjacent_to_stack(direction='right'|'left'|'front'|'back', strategy='tallest'|'nearest', clearance_mm=15.0)
- time.sleep(seconds)

ENHANCED UNDERSTANDING RULES:
1. Context references:
   - "there" or "same place" -> use system_state.last_place_location (x,y,z).
   - "that", "it", or "the previous block" -> use system_state.get_last_manipulated_block_id().

2. Stacks:
   - A stack means a location with >= 2 blocks already stacked.
   - "top of the stack" -> place on the tallest existing stack:
       (x, y, h) = system_state.get_tallest_stack_xy()
       z_offset = h * 17.9

3. Closest/Farthest:
   - Use nearest/farthest helper methods instead of guessing IDs. Distances are computed in Dobot XY space.

4. Automatic stack handling:
   - Before placing at (x, y), compute:
       h = system_state.get_stack_height(x, y)
       z_offset = h * 17.9
   - blocks_below = system_state.get_stack_at_location(x, y)
   - After placing, call:
       system_state.update_block_position(block_id, x, y, z_offset, blocks_below=blocks_below)
       system_state.update_gripper(None)

5. "Next to" placement (important):
   - When asked to place a block "next to" another block or stack, compute a safe target:
       a) system_state.compute_adjacent_to_block(target_id, direction, clearance_mm=15.0)
       b) or system_state.compute_adjacent_to_stack(direction='right', strategy='tallest', clearance_mm=15.0)
   - This guarantees >= 15 mm center-to-center clearance. Do NOT place closer.

6. Smart return behavior:
   - If the robot is already holding a different block and is asked to pick a new one,
     you must place the held block back at its original/current position first.

7. Color-only references:
   - "pick the red block" -> system_state.get_nearest_block_id_by_color("red") by default.

EXAMPLE A - Using last position:
USER: "place it at the last position you placed something"
STATE: Holding red_1, last_place_location = (300, 200, 0)
CODE:
target_x = system_state.last_place_location['x']
target_y = system_state.last_place_location['y']
stack_height = system_state.get_stack_height(target_x, target_y)
z_offset = stack_height * 17.9
dobot.place_block(target_x, target_y, z_offset)
stack_below = system_state.get_stack_at_location(target_x, target_y)
system_state.update_block_position('red_1', target_x, target_y, z_offset, blocks_below=stack_below)
system_state.update_gripper(None)

EXAMPLE B - Top of the tallest stack:
USER: "put the green block on top of the stack"
STATE: Gripper EMPTY
CODE:
green_id = system_state.get_nearest_block_id_by_color("green")
green = system_state.get_block(green_id)
dobot.pick_block(green['pixel_x'], green['pixel_y'])
system_state.update_gripper(green_id)
time.sleep(0.3)
sx, sy, h = system_state.get_tallest_stack_xy()
z = h * 17.9
dobot.place_block(sx, sy, z)
below = system_state.get_stack_at_location(sx, sy)
system_state.update_block_position(green_id, sx, sy, z, blocks_below=below)
system_state.update_gripper(None)

EXAMPLE C - Place right next to a block with safe clearance:
USER: "place it right next to blue_2"
STATE: Holding red_1
CODE:
nx, ny = system_state.compute_adjacent_to_block("blue_2", direction="right", clearance_mm=15.0)
h = system_state.get_stack_height(nx, ny)
z = h * 17.9
dobot.place_block(nx, ny, z)
below = system_state.get_stack_at_location(nx, ny)
system_state.update_block_position("red_1", nx, ny, z, blocks_below=below)
system_state.update_gripper(None)

CRITICAL: Always compute stack_height before placing so multi-level stacks are handled correctly.

Now, generate ONLY executable Python code for:
"""

    def generate_code(self, user_prompt, system_state):
        """Generate Python code from a natural language command."""
        state_summary = system_state.get_state_summary()
        system_prompt = self.system_prompt_template.format(state_summary=state_summary)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        try:
            print(f"\n[LLM] Analyzing: '{user_prompt}'")
            print(f"[LLM] Current gripper state: {system_state.gripper_status}")
            response = self.llm.invoke(messages)
            generated_code = response.content.strip()

            # Strip off any Markdown fences the LLM may have produced
            if "```python" in generated_code:
                generated_code = generated_code.split("```python")[1].split("```")[0].strip()
            elif "```" in generated_code:
                generated_code = generated_code.split("```")[1].split("```")[0].strip()

            print(f"[LLM] ✓ Code generated ({len(generated_code)} chars)")
            return generated_code
        except Exception as e:
            print(f"[LLM] ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return None


class CodeExecutor:
    """Validates and executes LLM-generated Python code in a controlled environment."""

    def __init__(self):
        # Only these calls are allowed to appear in the generated code
        self.allowed_calls = [
            'dobot.pick_block',
            'dobot.place_block',
            'system_state.update_gripper',
            'system_state.update_block_position',
            'system_state.get_block',
            'system_state.get_stack_at_location',
            'system_state.get_stack_height',
            'system_state.is_gripper_empty',
            'system_state.get_holding_block_id',
            'system_state.get_last_manipulated_block_id',
            'system_state.get_nearest_block_id_by_color',
            'system_state.get_farthest_block_id_by_color',
            'system_state.get_nearest_block_id',
            'system_state.get_farthest_block_id',
            'system_state.get_tallest_stack_xy',
            'system_state.get_nearest_stack_xy',
            'system_state.compute_adjacent_position',
            'system_state.compute_adjacent_to_block',
            'system_state.compute_adjacent_to_stack',
            'time.sleep'
        ]

    def validate_code(self, code_string):
        """
        Parse the generated code with AST and ensure all function calls & imports
        are in the whitelist.
        """
        try:
            tree = ast.parse(code_string)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        call_name = f"{ast.unparse(node.func.value)}.{node.func.attr}"
                        if call_name not in self.allowed_calls:
                            print(f"[Validator] ✗ Blocked call: {call_name}")
                            return False

                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.ImportFrom):
                        module = node.module
                    else:
                        module = node.names[0].name
                    if module not in ['time']:
                        print(f"[Validator] ✗ Blocked import: {module}")
                        return False

            print("[Validator] ✓ Code validated successfully")
            return True
        except SyntaxError as e:
            print(f"[Validator] ✗ Syntax error: {e}")
            return False
        except Exception as e:
            print(f"[Validator] ✗ Validation error: {e}")
            return False

    def execute_code(self, code_string, dobot, system_state):
        """Execute validated code with a very restricted global environment."""
        if not self.validate_code(code_string):
            print("[Executor] Code validation failed, NOT executing")
            return False

        safe_globals = {
            'dobot': dobot,
            'system_state': system_state,
            'time': time,
            '__builtins__': {
                'print': print,
                'len': len,
                'range': range,
                'int': int,
                'float': float,
                'str': str,
                'round': round,
            }
        }

        try:
            print("\n[Executor] Executing code...")
            print("-" * 70)
            exec(code_string, safe_globals)
            print("-" * 70)
            print("[Executor] ✓ Execution complete")
            return True
        except Exception as e:
            print(f"\n[Executor] ✗ EXECUTION ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False


class PickPlaceOrchestrator:
    """
    High-level coordinator that ties together:
    - Dobot controller
    - Block detector
    - System state
    - LLM code generator
    - Code executor
    - Interactive CLI
    """

    def __init__(self, calibration_mode="homography", calibration_file=CALIBRATION_JSON):
        self.dobot = DobotController(
            calibration_mode=calibration_mode,
            calibration_file=calibration_file
        )
        self.detector = BlockDetector()
        # Share converter between Dobot and SystemState so distances match
        self.system_state = SystemState(coord_converter=self.dobot.coord_converter)
        self.llm = LLMCodeGenerator(GROQ_API_KEY)
        self.executor = CodeExecutor()

    def _refresh_stack_meta(self):
        for loc_key in list(self.system_state.stack_map.keys()):
            self.system_state._update_stack_meta_for_key(loc_key)

    def auto_update_blocks(self):
        """
        Automatically capture block positions using the camera and update
        only the (x, y) coordinates in the saved state.
        """
        print("\n" + "=" * 70)
        print(" AUTO-UPDATING BLOCK POSITIONS...")
        print("=" * 70)

        self.dobot.go_home()
        time.sleep(0.5)

        captured_blocks = self.detector.auto_capture_blocks(display_duration=2.0)

        if captured_blocks and len(captured_blocks) > 0:
            print(f"\n[Camera] Detected {len(captured_blocks)} blocks")
            for block in captured_blocks:
                block_id = block['global_id']
                if block_id in self.system_state.blocks:
                    self.system_state.blocks[block_id]['pixel_x'] = block['pixel_x']
                    self.system_state.blocks[block_id]['pixel_y'] = block['pixel_y']
                else:
                    # New block detected that we didn't know about
                    self.system_state.blocks[block_id] = {
                        'color': block['color'],
                        'id': block['id'],
                        'pixel_x': block['pixel_x'],
                        'pixel_y': block['pixel_y'],
                        'original_x': block['pixel_x'],
                        'original_y': block['pixel_y'],
                        'current_position': {'x': block['pixel_x'], 'y': block['pixel_y'], 'z': 0},
                        'stack_level': 0,
                        'blocks_below': []
                    }

            self._refresh_stack_meta()
            self.system_state.save_state()
            print("\n[System] ✓ Block positions updated and saved")
        else:
            print("\n[Camera] No blocks detected (keeping previous positions)")

    def _connect_and_home(self, step_label):
        print(f"\n[{step_label}] DOBOT CONNECTION")
        print("-" * 70)
        if not self.dobot.connect():
            print("\n✗ FATAL: Dobot connection failed")
            sys.exit(1)

    def initialize(self):
        """Run through initial setup: load state, connect Dobot, detect blocks if needed."""
        print("\n" + "=" * 70)
        print(" ENHANCED SYSTEM INITIALIZATION")
        print("=" * 70)

        # Attempt to load previous state
        print("\n[STEP 0/3] LOADING PREVIOUS STATE")
        print("-" * 70)
        if self.system_state.load_state():
            print("\n✓ Previous state loaded successfully!")
            print(f"  - {len(self.system_state.blocks)} blocks remembered")
            print(f"  - {len(self.system_state.conversation_history)} commands in history")

            use_saved = input("\nUse saved state? (y/n): ").strip().lower()
            if use_saved == 'y':
                print("[System] Using saved state, skipping initial block detection")

                self._connect_and_home("STEP 1/2")

                print("\n[STEP 2/2] POSITIONING")
                print("-" * 70)
                self.dobot.go_home()

                print("\n" + "=" * 70)
                print("✓ INITIALIZATION COMPLETE (USING SAVED STATE)")
                print("=" * 70)
                return

        # Fresh run (or user said "no" to saved state)
        self._connect_and_home("STEP 1/3")

        print("\n[STEP 2/3] POSITIONING")
        print("-" * 70)
        self.dobot.go_home()
        time.sleep(1)

        print("\n[STEP 3/3] BLOCK DETECTION")
        print("-" * 70)
        detected_blocks = self.detector.detect_all_blocks_live()
        if detected_blocks is None or len(detected_blocks) == 0:
            print("\n✗ FATAL: No blocks detected")
            self.dobot.close()
            sys.exit(1)

        self.system_state.initialize_blocks(detected_blocks)

        print("\n" + "=" * 70)
        print("✓ INITIALIZATION COMPLETE")
        print("=" * 70)

    def _print_help(self):
        print("\n" + "=" * 70)
        print(" READY FOR COMMANDS - ENHANCED MODE")
        print("=" * 70)
        print("\n Features:")
        print("  • Persistent memory across sessions")
        print("  • Contextual understanding ('last position', 'there', 'that')")
        print("  • Automatic stack detection (tallest/nearest)")
        print("  • Natural language commands powered by LLM")
        print("  • Safe 'next to' placement with ≥15 mm clearance")
        print("\nExample commands:")
        print("  - 'pick up the red block'")
        print("  - 'place it next to the blue block'")
        print("  - 'put another block on top of that'")
        print("  - 'place the red block at the last position'")
        print("  - 'place it right next to the stack'")
        print("\nSpecial commands:")
        print("  - 'status'      - Show current state")
        print("  - 'history'     - Show command history")
        print("  - 'stacks'      - Show known stacks")
        print("  - 'home'        - Move Dobot to home")
        print("  - 'open camera' - Manual camera view + update positions")
        print("  - 'reset state' - Clear saved state on disk")
        print("  - 'mode'        - Switch calibration mode")
        print("  - 'quit'        - Exit program")
        print("=" * 70 + "\n")

    def run_interactive_loop(self):
        """Main loop for talking to the robot via natural language commands."""
        self._print_help()

        while True:
            try:
                user_prompt = input("\n[You] Enter command: ").strip()
                if not user_prompt:
                    continue

                lower_cmd = user_prompt.lower()

                # Exit command
                if lower_cmd in ['quit', 'exit', 'q']:
                    print("\n[System] Shutting down...")
                    break

                # Quick status printout
                if lower_cmd == 'status':
                    print("\n" + "=" * 70)
                    print(self.system_state.get_state_summary())
                    print("=" * 70)
                    continue

                # Print command history
                if lower_cmd == 'history':
                    print("\n" + "=" * 70)
                    print("COMMAND HISTORY:")
                    print("-" * 70)
                    for i, entry in enumerate(self.system_state.conversation_history, 1):
                        print(f"{i}. '{entry['prompt']}'")
                        print(f"   -> {entry['action']} at {entry['timestamp']}")
                        if entry.get('details'):
                            print(f"   Details: {entry['details']}")
                    print("=" * 70)
                    continue

                # Show stacks
                if lower_cmd == 'stacks':
                    print("\n" + "=" * 70)
                    print("CURRENT STACKS:")
                    print("-" * 70)
                    stacks = self.system_state.get_stack_locations(min_height=2)
                    if stacks:
                        for s in stacks:
                            print(f"  loc {s['loc_key']} @ ({s['x']},{s['y']}): height={s['height']}")
                    else:
                        print("  No stacks (height >= 2) detected")
                    print("=" * 70)
                    continue

                # Return Dobot to home
                if lower_cmd == 'home':
                    self.dobot.go_home()
                    continue

                # Manual camera update
                if lower_cmd == 'open camera':
                    self.dobot.go_home()
                    time.sleep(0.5)
                    captured_blocks = self.detector.detect_all_blocks_live()
                    if captured_blocks:
                        for block in captured_blocks:
                            block_id = block['global_id']
                            if block_id in self.system_state.blocks:
                                self.system_state.blocks[block_id]['pixel_x'] = block['pixel_x']
                                self.system_state.blocks[block_id]['pixel_y'] = block['pixel_y']
                        self._refresh_stack_meta()
                        self.system_state.save_state()
                        print(f"\n✓ Updated {len(captured_blocks)} blocks")
                    continue

                # Reset state on disk
                if lower_cmd == 'reset state':
                    confirm = input("Are you sure? This will delete saved state (y/n): ").strip().lower()
                    if confirm == 'y':
                        if STATE_FILE.exists():
                            STATE_FILE.unlink()
                            print("[System] ✓ State file deleted")
                        # Recreate SystemState but keep the same coordinate converter
                        self.system_state = SystemState(coord_converter=self.dobot.coord_converter)
                        print("[System] ✓ State reset - please re-detect blocks when ready")
                    continue

                # Switch calibration modes at runtime
                if lower_cmd.startswith('mode'):
                    print(f"\n[System] Current calibration mode: {self.dobot.coord_converter.mode}")
                    new_mode = input("Switch to mode ('homography' / 'affine', Enter to keep): ").strip().lower()
                    if new_mode in ('homography', 'affine'):
                        self.dobot.coord_converter.set_mode(new_mode)
                        # Keep SystemState in sync for distance calculations
                        self.system_state.coord_converter = self.dobot.coord_converter
                    else:
                        print("[System] No change to calibration mode")
                    continue

                # Not a special command: send it to the LLM
                generated_code = self.llm.generate_code(user_prompt, self.system_state)
                if generated_code is None:
                    print("[System] Failed to generate code from LLM")
                    continue

                # Show generated code
                print("\n" + "=" * 70)
                print("GENERATED CODE:")
                print("-" * 70)
                print(generated_code)
                print("=" * 70)

                # Operator must approve before anything runs on hardware
                confirm = input("\nExecute this code? (y/n/show): ").strip().lower()
                if confirm == 'show':
                    print("\n" + generated_code)
                    confirm = input("\nExecute? (y/n): ").strip().lower()
                if confirm != 'y':
                    print("[System] Execution cancelled")
                    continue

                success = self.executor.execute_code(generated_code, self.dobot, self.system_state)

                if success:
                    self.system_state.add_to_history(
                        user_prompt,
                        "Success",
                        details={'code_length': len(generated_code)}
                    )
                    print("\n✓ Command completed successfully")
                    # Refresh block positions from the camera
                    self.auto_update_blocks()
                else:
                    self.system_state.add_to_history(user_prompt, "Failed")
                    print("\n✗ Command failed")
                    self.dobot.go_home()

            except KeyboardInterrupt:
                print("\n\n[System] Interrupted by user (Ctrl+C)")
                break
            except Exception as e:
                print(f"\n[System] ✗ ERROR: {e}")
                import traceback
                traceback.print_exc()

        # Cleanup before exit
        print("\n[System] Cleaning up...")
        try:
            self.dobot.go_home()
        except Exception:
            pass
        self.dobot.close()
        print(f"\n✓ System shutdown complete, state saved at {STATE_FILE}")
        print("=" * 70)

    def run(self):
        """Main entry point for the orchestrator."""
        try:
            self.initialize()
            self.run_interactive_loop()
        except Exception as e:
            print(f"\n✗ FATAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            try:
                self.dobot.close()
            except Exception:
                pass


def main():
    print(" Enhanced Dobot system starting up.")
    print(f"\n State File: {STATE_FILE}")
    if STATE_FILE.exists():
        print("✓ Found existing state file (you'll be asked if you want to use it).")
    else:
        print("ℹ No saved state yet (a new one will be created).")

    if not GROQ_API_KEY:
        print("\n⚠ GROQ_API_KEY is not set. Natural-language commands will fail until you set it.")

    print("\n Configuration:")
    print(f"  Dobot port:   {DOBOT_PORT}")
    print(f"  Camera index: {CAMERA_INDEX}")
    print(f"  Block Height: {BLOCK_HEIGHT} mm")

    proceed = input("\n Start system? (y/n): ").strip().lower()
    if proceed != 'y':
        print("\nExiting without starting the system.")
        return

    print("\n Calibration modes:")
    print("  1) Homography from calibration.json")
    print("  2) Built-in 5-point affine calibration")
    print("  3) Homography from field_calib.yml (ArUco markers)")
    mode_choice = input("\nSelect calibration mode [1/2/3, default=1]: ").strip()

    calibration_mode = "homography"
    calibration_file = CALIBRATION_JSON
    if mode_choice == '2':
        calibration_mode = "affine"
    elif mode_choice == '3':
        calibration_mode = "homography"
        calibration_file = FIELD_CALIB_YML

    print(f"\n[System] Using '{calibration_mode}' calibration from '{calibration_file}'\n")

    orchestrator = PickPlaceOrchestrator(
        calibration_mode=calibration_mode,
        calibration_file=calibration_file
    )
    orchestrator.run()


if __name__ == "__main__":
    main()
