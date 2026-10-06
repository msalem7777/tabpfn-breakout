"""Six-state teacher bank and ten-input TabPFN adapter."""
import numpy as np
from breakout_pfn.ball_paddle_policy import features, FEATURES
from breakout_pfn.wall_features import model_features, MODEL_FEATURES


def read_state(env, include_bricks=False):
    if include_bricks:
        raise ValueError('This package supports the ball/paddle experiment only')
    return features(env)


def inputs(x, cfg):
    return model_features(x, cfg)


def load_bank(path, stride, include_bricks=False):
    if include_bricks:
        raise ValueError('This package supports the ball/paddle experiment only')
    from breakout_pfn.ball_paddle import load_context
    x, y, known = load_context(path, stride)
    with np.load(path, allow_pickle=False) as data:
        keep = data['frames'] % stride == 0
        episodes, frames = data['episodes'][keep], data['frames'][keep]
    return x, y, known, episodes, frames, 'ball-paddle-catch-v1'


def names(include_bricks=False):
    if include_bricks:
        raise ValueError('This package supports the ball/paddle experiment only')
    return FEATURES, MODEL_FEATURES
