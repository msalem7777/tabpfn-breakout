"""One hosted fit and prediction per frame; no local neural model or fallback."""
import numpy as np
from breakout_pfn.frame_models import create_model


def predict_client(args, x, y, state, report, factory=None):
    """A fresh estimator prevents a previous context from being reused by mistake.

    The service still owns its execution details. Do not claim local float32
    settings apply remotely. Never retry with a teacher or a copied label.
    """
    if args.backend != 'client':
        raise ValueError('Hosted prediction requires backend=client')
    model, settings = (factory or create_model)(args)
    report['model_settings'] = settings
    report['model_instances'] += 1
    model.fit(np.asarray(x, np.float32).copy(), np.asarray(y, np.float32).copy())
    report['fit_calls'] += 1
    output = np.asarray(model.predict(np.asarray(state, np.float32)[None, :].copy()), float).reshape(-1)
    report['prediction_calls'] += 1
    if output.shape != (1,) or not np.isfinite(output).all():
        raise ValueError('Invalid client prediction; no fallback action used')
    return float(output[0])


def smoke_test():
    from types import SimpleNamespace
    calls = []
    class Model:
        def fit(self, x, y): calls.append(('fit', len(x)))
        def predict(self, x): calls.append(('predict', x.shape)); return [.42]
    def factory(args): calls.append(('create',)); return Model(), {'backend': 'client'}
    report = dict(model_instances=0, fit_calls=0, prediction_calls=0)
    for n in (3, 5):
        assert predict_client(SimpleNamespace(backend='client'), np.zeros((n, 10)),
                              np.zeros(n), np.zeros(10), report, factory) == .42
    assert report['fit_calls'] == report['prediction_calls'] == report['model_instances'] == 2
    assert [c[1] for c in calls if c[0] == 'fit'] == [3, 5]
    print('Client context smoke test passed (no API requests)')


if __name__ == '__main__':
    smoke_test()
