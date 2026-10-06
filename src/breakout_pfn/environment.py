"""Atari Breakout adapter exposing a continuous target-paddle-position action.

The learner receives explicit ball kinematics when ALE exposes them. DINO is
reserved for brick-layout context. This module also inspects pixels for environment plumbing:

1. reading the paddle from ALE RAM (with vision fallback) so a normalized target ``x in [0, 1]`` can be executed;
2. verifying that Breakout is actually running after RESET/FIRE and after a
   missed ball, so paused frames are never silently collected as training data.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from breakout_pfn.vision import detect_ball_xy

FrameCallback = Callable[[np.ndarray], None]


@dataclass(slots=True, frozen=True)
class BallKinematics:
    """Explicit normalized ball state supplied to the learner."""

    x: float
    y: float
    dx: float
    dy: float
    visible: bool
    source: str


@dataclass(slots=True, frozen=True)
class CompletedEpisode:
    """One episode: one launched ball from a fresh brick wall."""

    episode_index: int
    paddle_hits: int
    score_gained: float
    ended_by: str


@dataclass(slots=True)
class ContinuousStepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]
    frames_executed: int
    paddle_x: float | None
    target_x: float
    native_lives: int | None
    missed_ball: bool
    misses: int
    relaunched: bool
    level_cleared: bool
    brick_units_remaining: int | None
    ball_in_play: bool
    ball: BallKinematics
    completed_episode: CompletedEpisode | None
    current_episode_hits: int
    current_episode_score: float
    paddle_hit: bool
    emulator_frame: int

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


@dataclass(slots=True)
class _LaunchResult:
    observation: np.ndarray
    info: dict[str, Any]
    reward: float
    frames_executed: int
    terminated: bool
    truncated: bool

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


def decode_breakout_ram_ball(ram: np.ndarray, frame_shape: tuple[int, ...]) -> tuple[float, float] | None:
    """Decode ALE Breakout ball coordinates from the 128-byte RAM state.

    Breakout stores horizontal and vertical ball state in bytes 99 and 101.
    The affine mapping converts those RAM values to ALE screen pixels before
    normalization, keeping ball x in the same [0, 1] coordinate system as the
    paddle target. A zero y byte denotes the waiting/no-ball state.
    """
    values = np.asarray(ram, dtype=np.uint8).reshape(-1)
    if values.size <= 101 or int(values[101]) == 0:
        return None
    h, w = int(frame_shape[0]), int(frame_shape[1])
    pixel_x = 0.999043 * float(values[99]) - 48.370898
    pixel_y = 0.993263 * float(values[101]) + 11.227841
    return (
        float(np.clip(pixel_x / max(w - 1, 1), 0.0, 1.0)),
        float(np.clip(pixel_y / max(h - 1, 1), 0.0, 1.0)),
    )


def _ball_from_info(info: dict[str, Any], frame_shape: tuple[int, ...]) -> tuple[float, float] | None:
    """Use explicit environment coordinates if an environment supplies them."""
    candidates = (
        ("ball_x", "ball_y"),
        ("ball_x_position", "ball_y_position"),
        ("ballX", "ballY"),
    )
    h, w = int(frame_shape[0]), int(frame_shape[1])
    for x_key, y_key in candidates:
        if x_key not in info or y_key not in info:
            continue
        try:
            x = float(np.asarray(info[x_key]).reshape(-1)[0])
            y = float(np.asarray(info[y_key]).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            continue
        if x > 1.0 or y > 1.0:
            x /= max(w - 1, 1)
            y /= max(h - 1, 1)
        return float(np.clip(x, 0.0, 1.0)), float(np.clip(y, 0.0, 1.0))
    return None


def decode_breakout_ram_paddle(ram: np.ndarray, frame_shape: tuple[int, ...]) -> float | None:
    """Decode the Breakout paddle center from ALE RAM byte 72.

    The affine mapping returns the paddle center in ALE screen-pixel coordinates;
    the result is normalized to the same [0, 1] horizontal coordinate system as
    the teacher target. RAM feedback is preferred over image segmentation because
    the latter jitters by a few pixels as the sprite changes.
    """
    values = np.asarray(ram, dtype=np.uint8).reshape(-1)
    if values.size <= 72:
        return None
    width = int(frame_shape[1])
    pixel_x = 1.005232 * float(values[72]) - 39.797062
    if pixel_x < 0.0 or pixel_x > float(width - 1):
        return None
    return float(pixel_x / max(width - 1, 1))


def detect_paddle_x(frame: np.ndarray) -> float | None:
    """Estimate paddle center from pixels for actuator feedback only.

    This coordinate may be supplied as an explicit control-state feature. Atari Breakout's
    paddle is a bright, low-screen horizontal component. Thresholding brightness
    rather than an exact RGB constant is robust to common ALE render variants.
    """
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] < 3:
        return None
    height, width = image.shape[:2]
    y0 = int(height * 0.72)
    y1 = int(height * 0.93)
    crop = image[y0:y1, :, :3]
    brightness = crop.max(axis=2)
    mask = brightness >= 170

    # Paddle is wider than the ball; score rows by the longest bright run.
    best: tuple[int, int, int] | None = None
    for row in mask:
        padded = np.pad(row.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        for start, end in zip(starts, ends, strict=False):
            length = int(end - start)
            if length >= 8 and (best is None or length > best[2]):
                best = (int(start), int(end), length)
    if best is None:
        return None
    center = 0.5 * (best[0] + best[1] - 1)
    return float(np.clip(center / max(width - 1, 1), 0.0, 1.0))


def playfield_motion_pixels(previous: np.ndarray, current: np.ndarray) -> int:
    """Count changed pixels in the ball/brick playfield, excluding paddle/score.

    ELI5: while the game is running, the ball changes at least a few pixels in
    the middle of the screen from frame to frame. A waiting-for-FIRE screen does
    not. The crop intentionally excludes the score strip and the paddle, so
    paddle motion cannot fool this health check.
    """
    before = np.asarray(previous)
    after = np.asarray(current)
    if before.shape != after.shape or before.ndim != 3 or before.shape[2] < 3:
        return 0

    height = before.shape[0]
    y0 = int(height * 0.10)
    y1 = int(height * 0.78)
    delta = np.max(
        np.abs(after[y0:y1, :, :3].astype(np.int16) - before[y0:y1, :, :3].astype(np.int16)),
        axis=2,
    )
    return int(np.count_nonzero(delta >= 30))


class BreakoutEnvironment:
    """Gymnasium/ALE Breakout with a normalized real-valued target-x action.

    The adapter owns Breakout's FIRE/start state machine. A lost ball is an
    episode boundary: it is returned to the trainer without relaunching so the
    trainer can perform a full reset and restore the initial brick wall.
    """

    def __init__(
        self,
        env_id: str,
        control_frames_per_decision: int = 2,
        target_tolerance: float = 1.0 / 159.0,
        launch_attempts: int = 3,
        launch_verify_frames: int = 12,
        launch_motion_min_pixels: int = 2,
        static_macro_steps_before_relaunch: int = 2,
        ball_position_source: str = "auto",
        miss_penalty: float = -1.0,
        safe_paddle_margin: float = 0.07,
        paddle_center_min_x: float = 15.5 / 159.0,
        paddle_center_max_x: float = 152.5 / 159.0,
        repeat_action_probability: float = 0.0,
        ball_velocity_window_frames: int = 5,
        servo_retarget_tolerance: float = 2.0 / 159.0,
        _env: Any | None = None,
    ):
        if _env is None:
            try:
                import gymnasium as gym
                import ale_py  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "Install gymnasium Atari/ALE dependencies to run Breakout."
                ) from exc
            self.env = gym.make(
                env_id,
                frameskip=1,
                repeat_action_probability=float(repeat_action_probability),
                render_mode="rgb_array",
            )
        else:
            # Dependency-injected environment used by unit tests. Keeping this
            # path here makes the FIRE/life-loss state machine testable without
            # requiring an Atari ROM in the test suite.
            self.env = _env

        self.control_frames_per_decision = int(control_frames_per_decision)
        self.target_tolerance = float(target_tolerance)
        self.launch_attempts = int(launch_attempts)
        self.launch_verify_frames = int(launch_verify_frames)
        self.launch_motion_min_pixels = int(launch_motion_min_pixels)
        self.static_macro_steps_before_relaunch = int(static_macro_steps_before_relaunch)
        if ball_position_source not in {"auto", "ram", "vision"}:
            raise ValueError("ball_position_source must be auto, ram, or vision")
        self.ball_position_source = str(ball_position_source)
        self.miss_penalty = float(miss_penalty)
        self.safe_paddle_margin = float(np.clip(safe_paddle_margin, 0.0, 0.499999))
        self.paddle_center_min_x = float(np.clip(paddle_center_min_x, 0.0, 1.0))
        self.paddle_center_max_x = float(np.clip(paddle_center_max_x, 0.0, 1.0))
        if self.paddle_center_min_x >= self.paddle_center_max_x:
            raise ValueError("invalid paddle-center bounds")
        self.repeat_action_probability = float(np.clip(repeat_action_probability, 0.0, 1.0))
        self.ball_velocity_window_frames = int(max(ball_velocity_window_frames, 2))
        self.servo_retarget_tolerance = float(max(servo_retarget_tolerance, self.target_tolerance))

        # Stateful paddle controller: release before arrival, observe coasting,
        # then make small pulses. Crossing a target does not mean the paddle has
        # stopped, and must never permanently disable position correction.
        self._servo_target_x: float | None = None
        self._servo_last_error: float | None = None
        self._servo_settled = False
        self._servo_previous_x: float | None = None
        self._servo_coasting = False
        self._servo_still_frames = 0
        self._paddle_width_pixels = 16

        meanings = list(self.env.unwrapped.get_action_meanings())
        self.noop_action = meanings.index("NOOP")
        self.fire_action = meanings.index("FIRE") if "FIRE" in meanings else self.noop_action
        self.left_action = meanings.index("LEFT")
        self.right_action = meanings.index("RIGHT")

        self.last_observation: np.ndarray | None = None
        self.last_info: dict[str, Any] = {}
        self._ball_verified = False
        self._static_macro_steps = 0
        self.relaunch_count = 0
        self.misses_current_level = 0
        self._previous_ball_xy: tuple[float, float] | None = None
        self._ball_history: deque[tuple[int, float, float]] = deque(maxlen=64)
        self._ball_state = BallKinematics(0.0, 0.0, 0.0, 0.0, False, "none")
        self._brick_units_initial: int | None = None
        self._brick_units_previous: int | None = None

        # One episode is one launched ball. Native Atari lives are used only to
        # detect a miss; every miss is followed by a full reset by the trainer.
        self._current_episode_hits = 0
        self._current_episode_score = 0.0
        self._hit_prev_xy: tuple[float, float] | None = None
        self._hit_prev_dy: float | None = None
        self.total_emulator_frames = 0


    def _reset_hit_tracker(self) -> None:
        self._hit_prev_xy = None
        self._hit_prev_dy = None

    def _track_paddle_hit(self, observation: np.ndarray, info: dict[str, Any]) -> bool:
        """Count a paddle contact from explicit ball kinematics, not score.

        Screen y increases downward. A paddle hit therefore appears as a change
        from positive dy (ball travelling down) to negative dy (travelling up)
        while the ball is in the lower playfield and horizontally near the paddle.
        This is deliberately independent of game score because paddle contacts do
        not themselves award points.
        """
        xy, _ = self._read_ball_xy(observation, info)
        if xy is None:
            self._reset_hit_tracker()
            return False

        x, y = xy
        if self._hit_prev_xy is None:
            self._hit_prev_xy = (x, y)
            return False

        dy = float(y - self._hit_prev_xy[1])
        hit = False
        if self._hit_prev_dy is not None:
            paddle_x = self._read_paddle_x(observation)
            near_paddle_y = max(y, self._hit_prev_xy[1]) >= 0.72
            near_paddle_x = paddle_x is None or abs(x - paddle_x) <= 0.16
            reversed_upward = self._hit_prev_dy > 1e-4 and dy < -1e-4
            hit = bool(near_paddle_y and near_paddle_x and reversed_upward)
            if hit:
                self._current_episode_hits += 1

        self._hit_prev_xy = (x, y)
        self._hit_prev_dy = dy
        return hit

    def _complete_current_episode(self, ended_by: str) -> CompletedEpisode:
        record = CompletedEpisode(
            episode_index=0,  # Trainer owns the single authoritative episode counter.
            paddle_hits=int(self._current_episode_hits),
            score_gained=float(self._current_episode_score),
            ended_by=str(ended_by),
        )
        self._current_episode_hits = 0
        self._current_episode_score = 0.0
        self._reset_hit_tracker()
        self._reset_servo()
        return record

    def _read_lives(self, info: dict[str, Any]) -> int | None:
        """Read native Atari lives. These are emulator plumbing, not episodes."""
        ram = self._read_ram()
        if ram is not None and ram.size > 57:
            return int(ram[57])
        try:
            return int(self.env.unwrapped.ale.lives())
        except (AttributeError, TypeError, ValueError):
            pass
        for key in ("lives", "ale.lives"):
            value = info.get(key)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass
        return None

    def _read_brick_units(self) -> int | None:
        """Return a monotone-ish RAM occupancy measure for the current brick field.

        Breakout RAM bytes 6..30 encode the brick map. Popcount is not presented
        as a literal brick count; it is only used to detect the large reset from
        an almost-cleared wall to the freshly populated next level.
        """
        ram = self._read_ram()
        if ram is None or ram.size <= 30:
            return None
        return int(np.unpackbits(ram[6:31]).sum())

    def _update_level_tracking(self) -> tuple[bool, int | None]:
        current = self._read_brick_units()
        if current is None:
            return False, None
        if self._brick_units_initial is None:
            self._brick_units_initial = current
            self._brick_units_previous = current
            return False, current
        previous = self._brick_units_previous
        initial = max(int(self._brick_units_initial), 1)
        level_cleared = False
        if previous is not None:
            # The ROM repopulates the brick RAM when advancing to the next wall.
            # Require both a nearly-cleared prior state and a large refill so a
            # small transient bit change cannot create a false level boundary.
            level_cleared = bool(
                previous <= max(8, int(0.20 * initial))
                and current >= max(16, int(0.70 * initial))
                and current > previous
            )
        self._brick_units_previous = current
        return level_cleared, current

    def begin_next_level(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Acknowledge a cleared wall without resetting the Atari game.

        The ROM is allowed to advance naturally to its next brick layout. If it
        parks the new ball waiting for FIRE, launch it; otherwise preserve the
        already-active ball. Crucially, ``env.reset()`` is not called here.
        """
        if self.last_observation is None:
            raise RuntimeError("Cannot begin the next level before reset().")
        self.misses_current_level = 0
        current = self._read_brick_units()
        self._brick_units_initial = current
        self._brick_units_previous = current

        xy, _ = self._read_ball_xy(self.last_observation, self.last_info)
        if xy is None:
            self._ball_verified = False
            launched = self._launch_and_verify(reason="new-level")
            if launched.done:
                raise RuntimeError("Breakout ended while launching the next cleared level.")
            observation = launched.observation
            info = launched.info
        else:
            observation = self.last_observation
            info = self.last_info.copy()
            self._ball_verified = True

        self._previous_ball_xy = None
        self._update_ball_state(observation, info, reset_velocity=True)
        return observation, info.copy()

    def _read_ram(self) -> np.ndarray | None:
        try:
            return np.asarray(self.env.unwrapped.ale.getRAM(), dtype=np.uint8).copy()
        except (AttributeError, TypeError, ValueError):
            return None

    def _read_paddle_x(self, observation: np.ndarray) -> float | None:
        """Return paddle center from RAM, falling back to pixels only if needed."""
        ram = self._read_ram()
        if ram is not None:
            # The real Breakout paddle shrinks from 16 to 12 pixels. RAM 72
            # locates its left edge, so a fixed affine center drifts by about two
            # pixels after shrinking. Measure the visible sprite width first.
            if observation.shape[:2] == (210, 160) and ram.size > 72:
                left = int(ram[72]) - 47
                if 8 <= left <= 136:
                    crop = observation[189:193, left:left + 16, :3]
                    red = (crop[..., 0] > 150) & (crop[..., 1] < 130) & (crop[..., 2] < 130)
                    width = int(np.count_nonzero(red.sum(axis=0) >= 3))
                    if width in {12, 16}:
                        self._paddle_width_pixels = width
                        return float((left + (width - 1) / 2.0) / 159.0)
            decoded = decode_breakout_ram_paddle(ram, observation.shape)
            if decoded is not None:
                return float(decoded - (16 - self._paddle_width_pixels) / (2.0 * max(observation.shape[1] - 1, 1)))
        return detect_paddle_x(observation)

    def paddle_x(self, observation: np.ndarray | None = None) -> float | None:
        """Public normalized paddle center using RAM-first feedback."""
        frame = self.last_observation if observation is None else np.asarray(observation)
        if frame is None:
            return None
        return self._read_paddle_x(frame)

    def _reset_servo(self) -> None:
        self._servo_target_x = None
        self._servo_last_error = None
        self._servo_settled = False
        self._servo_previous_x = None
        self._servo_coasting = False
        self._servo_still_frames = 0

    def _servo_action(self, requested_target: float, observation: np.ndarray) -> tuple[int, float | None]:
        """Choose movement using measured drift and short near-target pulses.

        The paddle keeps moving briefly after a stop command. Release early,
        wait for that motion to finish, then correct any remaining position error.
        """
        paddle_x = self._read_paddle_x(observation)
        if paddle_x is None:
            return self.noop_action, None

        requested = self.safe_target_x(requested_target)
        if (
            self._servo_target_x is None
            or abs(requested - self._servo_target_x) >= self.servo_retarget_tolerance
        ):
            self._servo_target_x = requested
            self._servo_last_error = None
            self._servo_settled = False

        target = float(self._servo_target_x)
        error = target - float(paddle_x)
        pixel = 1.0 / max(int(observation.shape[1]) - 1, 1)
        velocity = 0.0 if self._servo_previous_x is None else paddle_x - self._servo_previous_x
        self._servo_previous_x = float(paddle_x)
        self._servo_last_error = error
        self._servo_still_frames = self._servo_still_frames + 1 if abs(velocity) < 0.25 * pixel else 0

        # ALE's paddle has a one-frame response delay and coasts after NOOP.
        # Finish the coast before issuing another short correction. Two still
        # samples prevent the response delay from masquerading as a full stop.
        if self._servo_coasting:
            if self._servo_still_frames < 2:
                return self.noop_action, paddle_x
            self._servo_coasting = False

        # A single direction pulse moves roughly five pixels after settling.
        # Accept half that distance rather than bouncing between reachable spots.
        tolerance = max(self.target_tolerance, 2.75 * pixel)
        if abs(error) <= tolerance:
            self._servo_settled = self._servo_still_frames >= 2
            self._servo_coasting = not self._servo_settled
            return self.noop_action, paddle_x

        self._servo_settled = False
        moving_toward = velocity * error > 0.0
        # At full speed, release about two frame-displacements before the target.
        # The small allowance covers the last low-speed tail of the coast.
        if moving_toward and abs(error) <= 2.0 * abs(velocity) + 3.0 * pixel:
            self._servo_coasting = True
            self._servo_still_frames = 0
            return self.noop_action, paddle_x

        # Close to the target, issue one pulse and observe its complete effect.
        if abs(error) <= 18.0 * pixel:
            self._servo_coasting = True
            self._servo_still_frames = 0
        return (self.right_action if error > 0.0 else self.left_action), paddle_x

    def _read_ball_xy(
        self, observation: np.ndarray, info: dict[str, Any]
    ) -> tuple[tuple[float, float] | None, str]:
        explicit = _ball_from_info(info, observation.shape)
        if explicit is not None and self.ball_position_source in {"auto", "ram"}:
            return explicit, "info"

        if self.ball_position_source in {"auto", "ram"}:
            ram = self._read_ram()
            if ram is not None:
                decoded = decode_breakout_ram_ball(ram, observation.shape)
                if decoded is not None:
                    return decoded, "ale-ram"
            if self.ball_position_source == "ram":
                return None, "ram-unavailable"

        visual = detect_ball_xy(observation)
        if visual is not None:
            return visual, "vision-fallback"
        return None, "unavailable"

    def _update_ball_state(
        self,
        observation: np.ndarray,
        info: dict[str, Any],
        *,
        elapsed_frames: int = 1,
        reset_velocity: bool = False,
    ) -> BallKinematics:
        xy, source = self._read_ball_xy(observation, info)
        if xy is None:
            self._previous_ball_xy = None
            self._ball_history.clear()
            self._ball_state = BallKinematics(0.0, 0.0, 0.0, 0.0, False, source)
            return self._ball_state

        x, y = xy
        frame_index = int(self.total_emulator_frames)
        if reset_velocity:
            self._ball_history.clear()

        self._ball_history.append((frame_index, float(x), float(y)))
        cutoff = frame_index - self.ball_velocity_window_frames + 1
        samples = [row for row in self._ball_history if row[0] >= cutoff]

        if len(samples) < 2:
            dx = dy = 0.0
        else:
            # RAM coordinates are integer-valued. A one-frame finite difference
            # therefore alternates between 0 and +/-1 pixel even at constant
            # physical speed. Fit a line over several raw ALE frames instead.
            t = np.asarray([row[0] for row in samples], dtype=np.float64)
            xs = np.asarray([row[1] for row in samples], dtype=np.float64)
            ys = np.asarray([row[2] for row in samples], dtype=np.float64)
            t = t - float(np.mean(t))
            denom = float(np.dot(t, t))
            if denom <= 0.0:
                dx = dy = 0.0
            else:
                dx = float(np.dot(t, xs - float(np.mean(xs))) / denom)
                dy = float(np.dot(t, ys - float(np.mean(ys))) / denom)

        self._previous_ball_xy = (x, y)
        self._ball_state = BallKinematics(x, y, float(dx), float(dy), True, source)
        return self._ball_state

    @property
    def ball_state(self) -> BallKinematics:
        return self._ball_state

    def _launch_and_verify(
        self,
        reason: str,
        frame_callback: FrameCallback | None = None,
    ) -> _LaunchResult:
        """Press FIRE and refuse to continue until playfield motion is observed."""
        if self.last_observation is None:
            raise RuntimeError("Cannot launch Breakout before reset().")

        total_reward = 0.0
        total_frames = 0
        observation = self.last_observation
        info = self.last_info.copy()
        terminated = False
        truncated = False

        for attempt in range(1, self.launch_attempts + 1):
            observation, reward, terminated, truncated, info = self.env.step(self.fire_action)
            self.total_emulator_frames += 1
            observation = np.asarray(observation)
            total_reward += float(reward)
            total_frames += 1
            if frame_callback is not None:
                frame_callback(observation)
            if terminated or truncated:
                break

            previous = observation
            motion_hits = 0
            for _ in range(self.launch_verify_frames):
                observation, reward, terminated, truncated, info = self.env.step(self.noop_action)
                self.total_emulator_frames += 1
                observation = np.asarray(observation)
                total_reward += float(reward)
                total_frames += 1
                if frame_callback is not None:
                    frame_callback(observation)

                changed = playfield_motion_pixels(previous, observation)
                if changed >= self.launch_motion_min_pixels:
                    motion_hits += 1
                previous = observation

                if motion_hits >= 2:
                    self.last_observation = observation
                    self.last_info = dict(info)
                    self._ball_verified = True
                    self._static_macro_steps = 0
                    if reason != "reset":
                        self.relaunch_count += 1
                    lives = self._read_lives(self.last_info)
                    print(
                        f"[env] ball active reason={reason} attempt={attempt} lives={lives}",
                        flush=True,
                    )
                    return _LaunchResult(
                        observation=observation,
                        info=dict(info),
                        reward=total_reward,
                        frames_executed=total_frames,
                        terminated=False,
                        truncated=False,
                    )

                if terminated or truncated:
                    break

            if terminated or truncated:
                break

        self.last_observation = observation
        self.last_info = dict(info)
        self._ball_verified = False

        if terminated or truncated:
            return _LaunchResult(
                observation=observation,
                info=dict(info),
                reward=total_reward,
                frames_executed=total_frames,
                terminated=bool(terminated),
                truncated=bool(truncated),
            )

        raise RuntimeError(
            "Breakout FIRE was issued but no ball/playfield motion was detected. "
            "Learning has been stopped before paused frames could enter replay. "
            "Check the configured Atari environment/action meanings if this persists."
        )

    def safe_target_x(self, target_x: float) -> float:
        """Clamp a requested paddle center to the actually reachable safe corridor."""
        lo = max(self.safe_paddle_margin, self.paddle_center_min_x)
        hi = min(1.0 - self.safe_paddle_margin, self.paddle_center_max_x)
        if lo >= hi:
            raise RuntimeError("Configured paddle safety constraints leave no reachable corridor")
        return float(np.clip(target_x, lo, hi))

    def _position_paddle_before_launch(
        self, target_x: float, frame_callback: FrameCallback | None = None
    ) -> None:
        """Move the waiting paddle before FIRE to vary launch geometry.

        LEFT/RIGHT do not launch Breakout; FIRE is issued only after this helper
        returns. This keeps episode starts varied without adding sticky-action noise.
        """
        if self.last_observation is None:
            return
        target = self.safe_target_x(target_x)
        observation = self.last_observation
        info = self.last_info.copy()
        self._reset_servo()
        for _ in range(160):
            action, paddle_x = self._servo_action(target, observation)
            if self._servo_settled:
                break
            observation, _reward, terminated, truncated, info = self.env.step(action)
            self.total_emulator_frames += 1
            observation = np.asarray(observation)
            if frame_callback is not None:
                frame_callback(observation)
            if terminated or truncated:
                raise RuntimeError("Breakout terminated while randomizing the pre-launch paddle position.")
        self.last_observation = observation
        self.last_info = dict(info)

    def reset(
        self,
        seed: int | None = None,
        start_paddle_x: float | None = None,
        frame_callback: FrameCallback | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(seed=seed)
        self.last_observation = np.asarray(observation)
        self.last_info = dict(info)
        self._ball_verified = False
        self._static_macro_steps = 0
        self.misses_current_level = 0
        self._brick_units_initial = None
        self._brick_units_previous = None
        self._current_episode_hits = 0
        self._current_episode_score = 0.0
        self._reset_hit_tracker()

        self._paddle_width_pixels = 16
        self._reset_servo()
        if start_paddle_x is not None:
            self._position_paddle_before_launch(start_paddle_x, frame_callback=frame_callback)

        launched = self._launch_and_verify(reason="reset", frame_callback=frame_callback)
        if launched.done:
            raise RuntimeError("Breakout terminated while trying to launch after reset().")
        self._previous_ball_xy = None
        self._update_ball_state(
            launched.observation, launched.info, elapsed_frames=launched.frames_executed, reset_velocity=True
        )
        self._update_level_tracking()
        return launched.observation, launched.info.copy()

    def step_to_target(
        self,
        target_x: float,
        frame_callback: FrameCallback | None = None,
    ) -> ContinuousStepResult:
        if self.last_observation is None:
            raise RuntimeError("reset() must be called before step_to_target()")
        if not self._ball_verified:
            # This should normally be impossible within an episode because reset()
            # verifies motion. Keep the gate here as a final safety net.
            self._launch_and_verify(reason="pre-step-recovery", frame_callback=frame_callback)

        target = self.safe_target_x(target_x)
        total_reward = 0.0
        terminated = False
        truncated = False
        frames = 0
        observation = self.last_observation
        info = self.last_info.copy()
        lives_before = self._read_lives(info)
        missed_ball = False
        relaunched = False
        level_cleared = False
        brick_units_remaining = self._brick_units_previous
        motion_hits = 0
        completed_episode: CompletedEpisode | None = None
        paddle_hit = False

        for _ in range(self.control_frames_per_decision):
            action, paddle_x = self._servo_action(target, observation)

            previous = observation
            lives_before = self._read_lives(info)
            observation, reward, terminated, truncated, info = self.env.step(action)
            self.total_emulator_frames += 1
            observation = np.asarray(observation)
            raw_reward = float(reward)
            total_reward += raw_reward
            self._current_episode_score += raw_reward
            frames += 1
            # Velocity history must contain raw frames, even when a policy
            # decision advances two or more frames at a time.
            self._update_ball_state(observation, dict(info))
            paddle_hit = bool(self._track_paddle_hit(observation, dict(info)) or paddle_hit)
            if frame_callback is not None:
                frame_callback(observation)

            if playfield_motion_pixels(previous, observation) >= self.launch_motion_min_pixels:
                motion_hits += 1

            lives_after = self._read_lives(dict(info))
            if (
                lives_before is not None
                and lives_after is not None
                and lives_after < lives_before
                and not (terminated or truncated)
            ):
                missed_ball = True
                self.misses_current_level += 1
                completed_episode = self._complete_current_episode("miss")
                total_reward += self.miss_penalty
                self._ball_verified = False
                print(
                    "[env] ball lost; ending episode and resetting bricks before next ball",
                    flush=True,
                )
                # Do NOT FIRE again here. A miss is the episode boundary. The
                # trainer will call reset(), which restores the full brick wall
                # and launches the next episode from the same initial state.
                self.last_observation = observation
                self.last_info = dict(info)
                break

            level_cleared, brick_units_remaining = self._update_level_tracking()
            if level_cleared:
                completed_episode = self._complete_current_episode("level_clear")
                print(
                    f"[env] LEVEL CLEARED after misses={self.misses_current_level}",
                    flush=True,
                )
                break

            if terminated or truncated:
                self._ball_verified = False
                break


        self.last_observation = observation
        self.last_info = dict(info)

        # Fallback watchdog ONLY for environments where a reliable life counter
        # is unavailable.
        #
        # ALE/Breakout-v5 exposes the remaining lives directly. In that normal
        # case, a decrease in lives is the authoritative signal that the game is
        # waiting for FIRE again. Do not second-guess an active game from a
        # short pixel-motion window: the ball can legitimately spend several
        # decisions below the motion-check crop near the paddle, which made the
        # old watchdog falsely declare a stall and press FIRE during live play.
        #
        # The pixel watchdog remains useful for non-ALE/fallback environments
        # where no life counter is available at all.
        if not (terminated or truncated) and not missed_ball and not level_cleared:
            lives_now = self._read_lives(self.last_info)

            if lives_now is not None:
                # A valid ALE life counter means the dedicated life-loss branch
                # above owns all relaunch decisions. Preserve the verified state
                # even if this particular short window contained no detectable
                # ball pixels in the cropped motion region.
                self._static_macro_steps = 0
                self._ball_verified = True
            else:
                if motion_hits == 0:
                    self._static_macro_steps += 1
                else:
                    self._static_macro_steps = 0
                    self._ball_verified = True

                if self._static_macro_steps >= self.static_macro_steps_before_relaunch:
                    self._ball_verified = False
                    print(
                        "[env] no life counter and no playfield motion across consecutive "
                        "decisions; issuing FIRE recovery",
                        flush=True,
                    )
                    launched = self._launch_and_verify(
                        reason="stall-watchdog",
                        frame_callback=frame_callback,
                    )
                    total_reward += launched.reward
                    frames += launched.frames_executed
                    observation = launched.observation
                    info = launched.info
                    terminated = launched.terminated
                    truncated = launched.truncated
                    relaunched = not launched.done
                    self.last_observation = observation
                    self.last_info = dict(info)

        lives = self._read_lives(self.last_info)
        ball_in_play = bool(self._ball_verified and not (terminated or truncated))
        if missed_ball or relaunched or level_cleared or terminated or truncated:
            ball = self._update_ball_state(observation, self.last_info, reset_velocity=True)
        else:
            ball = self._ball_state
        return ContinuousStepResult(
            observation=observation,
            reward=total_reward,
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info),
            frames_executed=frames,
            paddle_x=self._read_paddle_x(observation),
            target_x=target,
            native_lives=lives,
            missed_ball=missed_ball,
            misses=self.misses_current_level,
            relaunched=relaunched,
            level_cleared=level_cleared,
            brick_units_remaining=brick_units_remaining,
            ball_in_play=ball_in_play,
            ball=ball,
            completed_episode=completed_episode,
            current_episode_hits=int(self._current_episode_hits),
            current_episode_score=float(self._current_episode_score),
            paddle_hit=bool(paddle_hit),
            emulator_frame=int(self.total_emulator_frames),
        )

    def close(self) -> None:
        self.env.close()

    def __enter__(self) -> "BreakoutEnvironment":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
