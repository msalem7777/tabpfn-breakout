"""Create, fit, predict once, and release one local estimator per frame."""
import gc
import sys
import numpy as np
from breakout_pfn.frame_models import create_model, model_settings


def release_cuda_cache():
    """Release unreachable objects and unused PyTorch blocks between models."""
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def predict_fresh(args, x, y, state, report, model_factory=None,
                  factory=create_model, cleanup=release_cuda_cache):
    """No estimator survives in this function between successful calls.

    Shared library/process state may survive. This is model reinstantiation,
    not process isolation. The existing factory installs the GPU cap each time.
    """
    if getattr(args, 'backend', 'client') != 'local':
        raise ValueError('Fresh-per-frame mode requires --backend local')
    model = None
    try:
        # Fresh settings prevent runtime metadata from a prior model from
        # accidentally being passed back as estimator constructor arguments.
        settings = model_settings(args)
        report['model_settings'] = settings
        model, settings = factory(args, model_factory=model_factory, settings=settings)
        report['model_settings'] = settings
        report['model_instances'] += 1
        model.fit(x.copy(), y.copy())
        report['fit_calls'] += 1
        prediction = np.asarray(model.predict(np.asarray(state, np.float32)[None, :].copy()),
                                dtype=float).reshape(-1)
        report['prediction_calls'] += 1
        if prediction.shape != (1,) or not np.isfinite(prediction).all():
            raise ValueError('Fresh model returned an invalid coordinate; no fallback action taken')
        return float(prediction[0])
    finally:
        # Drop our reference before garbage collection / empty_cache. Never
        # load the next model until this cleanup has completed.
        model = None
        if sys.exc_info()[0] is None:
            cleanup()
        else:
            try:
                cleanup()
            except Exception:
                pass  # Preserve the original fit/predict failure.


def smoke_test():
    from types import SimpleNamespace
    calls = []
    class Model:
        def fit(self, x, y): calls.append('fit')
        def predict(self, x):
            calls.append('predict')
            return [.4]
    def factory(args, **kwargs):
        calls.append('create')
        return Model(), kwargs['settings']
    args = SimpleNamespace(backend='local', model_path='v3.5_default')
    report = dict(model_instances=0, fit_calls=0, prediction_calls=0)
    for _ in range(2):
        assert predict_fresh(args, np.zeros((1, 114)), np.array([.4]), np.zeros(114),
                             report, factory=factory, cleanup=lambda: calls.append('release')) == .4
    assert calls == ['create', 'fit', 'predict', 'release']*2
    assert report['model_instances'] == report['fit_calls'] == report['prediction_calls'] == 2
    print('Fresh-per-frame smoke test passed (no GPU model loaded)')


if __name__ == '__main__':
    smoke_test()
