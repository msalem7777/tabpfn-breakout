"""A stateless, six-input catching teacher; no brick or simulator access."""
import numpy as np

FEATURES = ('ball_x', 'ball_y', 'ball_vx', 'ball_vy', 'paddle_x', 'paddle_vx')
POLICY_VERSION = 'ball-paddle-catch-v1'


def features(env):
    """Read current motion only. Do not compute a brick bitmap then discard it."""
    ball = env.ball_state
    paddle = env.paddle_x()
    previous = env._servo_previous_x
    return validate([ball.x, ball.y, ball.dx, ball.dy, paddle,
                     0.0 if previous is None else paddle - previous])


def validate(state):
    state = np.asarray(state, dtype=np.float32).copy()
    if state.shape != (6,) or not np.isfinite(state).all():
        raise ValueError('Expected six finite ball/paddle inputs')
    state[state == 0] = 0  # Give positive and negative zero one unique key.
    return state


def coordinate(state, cfg):
    """Predict a center catch from straight flight and side-wall reflections.

    Downward: project to paddle height and fold side-wall bounces back into
    the field. Upward: follow current ball x; zero vertical speed: hold paddle.
    Recompute every frame, so observed collisions immediately change the next
    target. This function cannot anticipate brick collisions or optimize score.
    """
    x, y, vx, vy, paddle, _ = map(float, validate(state))
    lo = max(cfg.safe_paddle_margin, cfg.paddle_center_min_x)
    hi = min(1.0 - cfg.safe_paddle_margin, cfg.paddle_center_max_x)
    left, right = cfg.ball_field_left_x, cfg.ball_field_right_x
    if not lo < hi or not left < right:
        raise ValueError('Invalid paddle or wall geometry')
    if vy > 0:
        frames = max(0.0, (cfg.teacher_paddle_y - y) / vy)
        width = right - left
        folded = (x + vx * frames - left) % (2.0 * width)
        target = left + (folded if folded <= width else 2.0 * width - folded)
    elif vy < 0:
        target = x
    else:
        target = paddle
    return float(np.clip(target, lo, hi))


def smoke_test():
    from breakout_pfn.config import ExperimentConfig
    cfg = ExperimentConfig()
    cfg.ball_field_left_x, cfg.ball_field_right_x = 0.0, 1.0
    cfg.teacher_paddle_y = 0.9
    assert abs(coordinate([.4, .5, .01, .02, .5, 0], cfg) - .6) < 1e-6
    assert abs(coordinate([.9, .5, .02, .02, .5, 0], cfg) - .7) < 1e-6
    assert abs(coordinate([.3, .5, .01, -.02, .5, 0], cfg) - .3) < 1e-6
    assert coordinate([.3, .5, 0, 0, .5, 0], cfg) == .5
    print('Ball/paddle policy smoke test passed (no game or model loaded)')


if __name__ == '__main__':
    smoke_test()
