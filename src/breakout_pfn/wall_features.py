"""Shared input/target transformations; no teacher or simulator calls."""
import numpy as np
from breakout_pfn.ball_paddle_policy import FEATURES

MODEL_FEATURES = FEATURES + ('ball_distance_left', 'ball_distance_right',
                             'paddle_distance_left', 'paddle_distance_right')


def model_features(states, cfg):
    """Append signed center distances to the configured side-wall coordinates.

    Units match normalized x. A negative distance means the measured center is
    beyond that configured boundary; preserve it rather than hiding it by clipping.
    Paddle distances use its center, not its changing half-width.
    """
    x = np.asarray(states, np.float32)
    if x.ndim not in (1, 2) or x.shape[-1] != 6 or not np.isfinite(x).all():
        raise ValueError('Wall features require finite six-column base states')
    left, right = cfg.ball_field_left_x, cfg.ball_field_right_x
    if not left < right:
        raise ValueError('Invalid side-wall coordinates')
    extra = np.stack((x[..., 0]-left, right-x[..., 0],
                      x[..., 4]-left, right-x[..., 4]), axis=-1)
    return np.concatenate((x, extra), axis=-1).astype(np.float32)


def model_targets(x, y, mode):
    """Optionally learn movement relative to the current paddle center."""
    if mode == 'absolute':
        return np.asarray(y, np.float32).copy()
    if mode == 'displacement':
        return np.asarray(y, np.float32) - np.asarray(x, np.float32)[:, 4]
    raise ValueError('Unknown prediction target')


def destination(prediction, state, mode):
    if mode not in ('absolute', 'displacement'):
        raise ValueError('Unknown prediction target')
    return float(prediction) + (float(state[4]) if mode == 'displacement' else 0.0)


def smoke_test():
    from breakout_pfn.config import ExperimentConfig
    cfg = ExperimentConfig()
    x = np.asarray([[.3, .4, .01, .02, .5, 0]], np.float32)
    z = model_features(x, cfg)
    assert z.shape == (1, 10)
    np.testing.assert_allclose(z[:, 6]+z[:, 7], cfg.ball_field_right_x-cfg.ball_field_left_x)
    np.testing.assert_allclose(z[:, 8]+z[:, 9], cfg.ball_field_right_x-cfg.ball_field_left_x)
    y = np.asarray([.4], np.float32)
    assert abs(destination(model_targets(x, y, 'displacement')[0], x[0], 'displacement')-.4) < 1e-6
    print('Wall feature smoke test passed (no game/model loaded)')


if __name__ == '__main__':
    smoke_test()
