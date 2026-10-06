"""Native ALE variable-strength paddle input for precise target following.

No RAM writes, teleportation, ball/brick inspection or simulated futures. The
ROM still determines motion, delay, pixel quantization and collisions.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PaddleCommand:
    direction: int  # -1 left, 0 release, +1 right
    strength: float


def install(trace, env):
    """Install on one environment. Integer reset/FIRE actions keep their path."""
    base = env.env.unwrapped
    if not hasattr(base, 'continuous') or not hasattr(base, 'continuous_action_threshold'):
        raise RuntimeError('Analog tracking requires ALE Gymnasium continuous-action support')
    if hasattr(base, '_breakout_analog_installed'):
        raise RuntimeError('Analog adapter already installed on this environment')
    original_step = base.step

    def step(action):
        if not isinstance(action, PaddleCommand):
            return original_step(action)
        old_continuous, old_threshold = base.continuous, base.continuous_action_threshold
        try:
            base.continuous = True
            base.continuous_action_threshold = 1e-6
            angle = np.pi if action.direction < 0 else 0.0
            magnitude = action.strength if action.direction else 0.0
            return original_step(np.asarray([magnitude, angle, 0.0], np.float32))
        finally:
            base.continuous, base.continuous_action_threshold = old_continuous, old_threshold

    base.step = step
    base._breakout_analog_installed = True
    trace.last = None

    def choose(requested, observation):
        paddle = env._read_paddle_x(observation)
        if paddle is None:
            raise RuntimeError('Cannot track an unobserved paddle')
        scale = max(observation.shape[1]-1, 1)
        target = env.safe_target_x(requested)
        previous = env._servo_target_x
        # Accept EVERY requested coordinate; never keep a stale nearby target.
        env._servo_target_x = target
        velocity = 0.0 if env._servo_previous_x is None else (paddle-env._servo_previous_x)*scale
        env._servo_previous_x = float(paddle)
        env._servo_still_frames = env._servo_still_frames+1 if abs(velocity) < .25 else 0
        error = (target-paddle)*scale
        command = decide(error, velocity, env._servo_coasting, env._servo_still_frames)
        action, coasting = command
        env._servo_coasting = coasting
        if action.direction and abs(error) <= 18:
            env._servo_still_frames = 0
        env._servo_settled = abs(error) <= .5 and env._servo_still_frames >= 2
        trace.last = dict(action='analog', direction=action.direction, strength=action.strength,
                          requested_target=float(requested), accepted_target=target,
                          previous_accepted_target=previous, paddle_x=float(paddle),
                          error_pixels=error, velocity_pixels=velocity,
                          coasting=coasting, still_frames=env._servo_still_frames,
                          tolerance_pixels=.5)
        return action, paddle

    env._servo_action = choose


def decide(error, velocity, coasting, still_frames):
    """Full speed far away, fractional pulses near the requested coordinate.

    Native full pulses move about five pixels after settling in this ROM.
    Fractional strength allows one-pixel corrections. Wait for pending movement
    before issuing another close-range correction, avoiding double commands.
    """
    release = PaddleCommand(0, 0.0)
    if coasting and still_frames < 2:
        return release, True
    if abs(error) <= .5:
        return release, still_frames < 2
    if velocity*error > 0 and abs(error) <= 2*abs(velocity)+3:
        return release, True
    direction = 1 if error > 0 else -1
    if abs(error) <= 18:
        return PaddleCommand(direction, float(np.clip(abs(error)/5.0, .1, 1.0))), True
    return PaddleCommand(direction, 1.0), False


def smoke_test():
    assert decide(.4, 0, False, 3)[0].direction == 0
    assert decide(1, 0, False, 3)[0] == PaddleCommand(1, .2)
    assert decide(-1, 0, False, 3)[0] == PaddleCommand(-1, .2)
    assert decide(50, 0, False, 3)[0] == PaddleCommand(1, 1)
    assert decide(1, 1, True, 0)[0].direction == 0
    print('Analog paddle smoke test passed (no emulator/model loaded)')


if __name__ == '__main__':
    smoke_test()
