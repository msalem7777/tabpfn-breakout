"""Public TabPFN factories for the fixed-context frame runner.

Local predictions do not call the hosted client. No device shuttling, private
attention overrides, or per-prediction cache mutations are installed here.
An explicit PyTorch allocation ceiling is installed before local model loading.
"""
from importlib.metadata import PackageNotFoundError, version
from breakout_pfn.gpu_budget import install_budget


def model_settings(args):
    backend = getattr(args, 'backend', 'client')
    if backend not in ('client', 'local'):
        raise ValueError(f'Unknown backend: {backend}')
    settings = dict(backend=backend, model_path=args.model_path,
                    fit_mode=getattr(args, 'client_fit_mode', 'fit_with_cache'))
    if backend == 'client' and hasattr(args, 'client_n_estimators'):
        if args.client_n_estimators < 1:
            raise ValueError('Client estimator count must be positive')
        settings.update(n_estimators=args.client_n_estimators,
                        random_state=getattr(args, 'model_seed', 7))
    if backend == 'local':
        settings.update(device='cuda',
                        fit_mode=getattr(args, 'local_fit_mode', 'fit_preprocessors'),
                        n_estimators=getattr(args, 'local_n_estimators', 1),
                        memory_saving_mode=True,
                        inference_precision=getattr(args, 'local_precision', 'auto'),
                        random_state=getattr(args, 'model_seed', 7),
                        show_progress_bar=False, n_preprocessing_jobs=1)
        if settings['n_estimators'] < 1:
            raise ValueError('Local estimator count must be positive')
        if settings['inference_precision'] not in ('auto', 'float32'):
            raise ValueError('Local precision must be auto or float32')
    return settings


def create_model(args, model_factory=None, settings=None):
    """Return a single estimator and the effective, logged configuration."""
    settings = model_settings(args) if settings is None else settings
    if settings['backend'] == 'client':
        if model_factory is None:
            from tabpfn_client import TabPFNRegressor
            model_factory = TabPFNRegressor
        kwargs = {k: v for k, v in settings.items() if k not in ('backend', 'model_path')}
        model = model_factory(model_path=args.model_path, **kwargs)
    else:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Local GPU mode requires CUDA-enabled PyTorch and an available '
                               'NVIDIA GPU. No client or CPU fallback was used.')
        # Install BEFORE importing/constructing TabPFN. Preserve the effective
        # budget in the caller's report even if model construction runs out.
        settings['gpu_budget'] = install_budget(
            torch, getattr(args, 'gpu_memory_gib', 5.0), getattr(args, 'gpu_headroom_gib', 1.0))
        try:
            from tabpfn import TabPFNRegressor
            from tabpfn.constants import ModelVersion
        except ImportError as exc:
            raise RuntimeError('Install local TabPFN with: python -m pip install -e ".[local]"') from exc
        factory = TabPFNRegressor if model_factory is None else model_factory
        kwargs = {k: v for k, v in settings.items()
                  if k not in ('backend', 'model_path', 'gpu_budget')}
        # TabPFN expects a torch.dtype for forced precision, not the CLI string.
        # Keep the logged settings JSON-serializable; change only constructor kwargs.
        if kwargs['inference_precision'] == 'float32':
            kwargs['inference_precision'] = torch.float32
        aliases = {'v3.5_default': 'V3_5', 'v3.5': 'V3_5', 'v3.5_fast': 'V3_5_FAST'}
        alias = aliases.get(args.model_path.lower())
        if alias is None:
            model = factory(model_path=args.model_path, **kwargs)
        else:
            model = factory.create_default_for_version(getattr(ModelVersion, alias), **kwargs)
        settings.update(torch_version=torch.__version__, gpu_name=torch.cuda.get_device_name())
    package = 'tabpfn' if settings['backend'] == 'local' else 'tabpfn-client'
    try:
        settings['package_version'] = version(package)
    except PackageNotFoundError:
        settings['package_version'] = 'unknown'
    return model, settings


def smoke_test():
    """Check configuration without importing TabPFN, downloading weights, or GPU use."""
    from types import SimpleNamespace
    args = SimpleNamespace(backend='local', model_path='v3.5_default')
    settings = model_settings(args)
    assert settings['device'] == 'cuda' and settings['n_estimators'] == 1
    assert settings['fit_mode'] == 'fit_preprocessors'
    args.backend = 'client'
    assert model_settings(args)['fit_mode'] == 'fit_with_cache'
    print('Frame model settings smoke test passed (no model loaded)')


if __name__ == '__main__':
    smoke_test()
