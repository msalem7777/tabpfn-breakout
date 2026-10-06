"""Shared game construction, outcome measurement and JSON logging."""
import json
from pathlib import Path
import numpy as np
from breakout_pfn.environment import BreakoutEnvironment

def environment(cfg):
    # Match the proven teacher profile's geometry and controller configuration.
    return BreakoutEnvironment(cfg.env_id, control_frames_per_decision=1,
        target_tolerance=cfg.target_tolerance, servo_retarget_tolerance=cfg.servo_retarget_tolerance,
        safe_paddle_margin=cfg.safe_paddle_margin,
        paddle_center_min_x=cfg.paddle_center_min_x, paddle_center_max_x=cfg.paddle_center_max_x,
        ball_position_source=cfg.ball_position_source,
        ball_velocity_window_frames=cfg.ball_velocity_window_frames, repeat_action_probability=0.)

def brick_layout(frame: np.ndarray) -> np.ndarray:
    """Read the 6 by 18 brick grid in an unscaled 210 by 160 ALE frame.

    A ball occupies fewer than half of a brick cell, so it cannot create a brick.
    This precise bitmap is small enough to give directly to TabPFN without DINO.
    """
    if frame.shape[:2] != (210, 160):
        raise ValueError("Brick layout requires the raw 210x160 Breakout image")
    occupied = np.max(frame[57:93, 8:152, :3], axis=2) > 100
    return (occupied.reshape(6, 6, 18, 8).mean(axis=(1, 3)) > 0.5).astype(np.float32)

def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def emit(file, record):
    file.write(json.dumps(record, allow_nan=False) + '\n')
    file.flush()
