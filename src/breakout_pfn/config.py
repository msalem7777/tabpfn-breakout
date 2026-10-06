"""Game settings; old recording profiles remain readable."""
import json
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Literal

@dataclass
class ExperimentConfig:
    env_id: str = "ALE/Breakout-v5"
    seed: int = 7
    target_tolerance: float = 1.0 / 159.0
    servo_retarget_tolerance: float = 2.0 / 159.0
    paddle_center_min_x: float = 15.5 / 159.0
    paddle_center_max_x: float = 152.5 / 159.0
    safe_paddle_margin: float = 0.07
    ball_field_left_x: float = 8.0 / 159.0
    ball_field_right_x: float = 151.0 / 159.0
    ball_velocity_window_frames: int = 5
    episode_start_paddle_min_x: float = 0.15
    episode_start_paddle_max_x: float = 0.85
    ball_position_source: str = "auto"
    teacher_paddle_y: float = 188.0 / 209.0

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2) + '\n', encoding='utf-8')

    @classmethod
    def load(cls, path):
        # Historical profiles include settings for experiments omitted here.
        allowed = {field.name for field in fields(cls)}
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(**{key: value for key, value in data.items() if key in allowed})
