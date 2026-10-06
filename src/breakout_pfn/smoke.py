"""Small offline checks: no model download, GPU allocation or client request."""
from importlib import import_module
from pathlib import Path
import numpy as np


def main():
    for name in ('ball_paddle', 'local_teacher_context', 'student_controller',
                 'frame_models', 'fresh_frame_model', 'gpu_budget', 'client_context',
                 'run_paths', 'decision_video'):
        import_module('breakout_pfn.' + name).smoke_test()
    from breakout_pfn.ball_paddle import load_context, fingerprint
    from breakout_pfn.config import ExperimentConfig
    from breakout_pfn.local_teacher_context import LocalTeacherContext
    from breakout_pfn.wall_features import model_features
    # Run from the repository root to verify the bundled demonstration bank.
    data = Path('data/teacher')
    x, y, _ = load_context(data/'demonstrations.npz', 1)
    assert len(x) == 23917
    assert fingerprint(x, y) == 'b7d6d45763fd6f9f466e04a2f68845a12e96eeb4371794d4944f155a023c7b68'
    with np.load(data/'demonstrations.npz', allow_pickle=False) as bank:
        selector = LocalTeacherContext(x, bank['episodes'], bank['frames'], velocity_tolerance_pixels=.05)
    ids, _ = selector.select(x[100])
    assert len(ids) == len(set(ids)) and 2 <= len(ids) <= 256
    assert model_features(x[ids], ExperimentConfig.load(data/'config.json')).shape == (len(ids), 10)
    print('Bundled teacher data and ten-input context checks passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
