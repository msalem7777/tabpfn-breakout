"""Collect and imitate the separate, brick-blind catching teacher."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np

from breakout_pfn.ball_paddle_policy import FEATURES, POLICY_VERSION, coordinate, features, validate
from breakout_pfn.config import ExperimentConfig
from breakout_pfn.runtime import environment
from breakout_pfn.run_paths import open_unique_run
from breakout_pfn.frame_models import create_model, model_settings
from breakout_pfn.fresh_frame_model import predict_fresh
from breakout_pfn.runtime import brick_layout
from breakout_pfn.runtime import emit, write_json
from breakout_pfn.student_controller import ControllerTrace
from breakout_pfn.local_teacher_context import LocalTeacherContext
from breakout_pfn.wall_features import MODEL_FEATURES, model_features, model_targets, destination
from breakout_pfn.client_context import predict_client
from breakout_pfn import experiment_inputs

SCHEMA = 'ball-paddle-six-input-v1'


def fingerprint(x, y):
    return hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()


class Rows:
    """Unique six-column inputs, labelled only after their action was executed."""
    def __init__(self):
        self.rows, self.keys = [], {}
        self.duplicates = 0

    def add(self, state, target, episode, frame):
        state = validate(state)
        target = np.float32(target)
        if not np.isfinite(target):
            raise ValueError('Nonfinite teacher target')
        key = state.tobytes()
        if key in self.keys:
            self.duplicates += 1
            if self.rows[self.keys[key]][1] != target:
                raise ValueError('Identical six-input state has conflicting teacher labels')
            return False
        self.keys[key] = len(self.rows)
        self.rows.append((state, target, episode, frame))
        return True

    def save(self, folder, count, cfg=None):
        # Keep first occurrence metadata; select evenly through unique history.
        # Never duplicate rows to meet a requested count.
        indices = (np.arange(len(self.rows)) if count == 0 else
                   np.linspace(0, len(self.rows)-1, min(count, len(self.rows)), dtype=int))
        rows = [self.rows[i] for i in indices]
        x = np.stack([r[0] for r in rows]) if rows else np.empty((0, 6), np.float32)
        y = np.asarray([r[1] for r in rows], np.float32)
        temporary = folder / 'demonstrations.tmp.npz'
        extra = {} if cfg is None else dict(model_X=model_features(x, cfg),
                                            model_feature_names=np.asarray(MODEL_FEATURES))
        np.savez_compressed(temporary, schema=SCHEMA, policy_version=POLICY_VERSION,
                            feature_names=np.asarray(FEATURES), X=x, y=y,
                            label_sources=np.full(len(rows), 'executed_teacher'),
                            episodes=np.asarray([r[2] for r in rows], dtype=np.int64),
                            frames=np.asarray([r[3] for r in rows], dtype=np.int64), **extra)
        temporary.replace(folder / 'demonstrations.npz')
        return len(rows)


def load_context(path, stride):
    if stride < 1:
        raise ValueError('Context stride must be positive')
    with np.load(path, allow_pickle=False) as data:
        if data['schema'].item() != SCHEMA or data['policy_version'].item() != POLICY_VERSION:
            raise ValueError('Collect new ball/paddle teacher rows; old labels are incompatible')
        if tuple(data['feature_names']) != FEATURES:
            raise ValueError('Unexpected feature order')
        x, y, frames = data['X'].copy(), data['y'].copy(), data['frames'].copy()
    if x.ndim != 2 or x.shape[1] != 6 or y.shape != (len(x),) or frames.shape != (len(x),):
        raise ValueError('Malformed six-input dataset')
    keep = frames % stride == 0
    x, y = x[keep], y[keep]
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Context must be nonempty and finite')
    keys = [validate(row).tobytes() for row in x]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate six-input context rows')
    return x, y, dict(zip(keys, y))


def outcome(result, bricks):
    # Bricks are measured ONLY for reporting and termination, never policy input.
    if result.level_cleared or bricks == 0:
        return 'level_clear'
    if result.missed_ball:
        return 'miss'
    return 'terminated' if result.done else None


def setup(args):
    source = Path(args.source)
    cfg = ExperimentConfig.load(source / 'config.json')
    episodes = json.loads((source / 'episodes.json').read_text())
    return source, cfg, episodes


def collect(args):
    _, cfg, starts = setup(args)
    if len(starts) < args.episodes:
        # Preserve all recorded starts; additional episodes use reproducible
        # new seeds and paddle positions within the configured launch interval.
        seed = getattr(args, 'collection_seed', 10007)
        rng = np.random.default_rng(seed)
        used = {int(s['seed']) for s in starts}
        while len(starts) < args.episodes:
            while seed in used:
                seed += 1
            starts.append(dict(seed=seed, start_paddle_x=float(rng.uniform(
                cfg.episode_start_paddle_min_x, cfg.episode_start_paddle_max_x))))
            used.add(seed)
            seed += 1
    folder = open_unique_run(args.run_dir)
    args.run_dir = str(folder)  # Persist the actual path in experiment.json.
    cfg.save(folder / 'config.json')
    write_json(folder / 'experiment.json', vars(args))
    pool, episodes = Rows(), []
    report = dict(policy_version=POLICY_VERSION, features=FEATURES,
                  model_features=MODEL_FEATURES,
                  controller=args.student_controller, status='running')
    try:
        with environment(cfg) as env, (folder / 'executed_frames.jsonl').open('w') as log:
            # Both teacher and student use exactly the same movement controller.
            controller = ControllerTrace(env, args.student_controller)
            for episode, start in enumerate(starts[:args.episodes]):
                env.reset(seed=start['seed'], start_paddle_x=start['start_paddle_x'])
                initial = features(env)
                score = hits = 0
                event = 'frame_limit'
                for frame in range(args.episode_frames):
                    state = features(env)
                    target = coordinate(state, cfg)
                    visible = bool(env.ball_state.visible)
                    result = env.step_to_target(target)
                    if result.frames_executed != 1:
                        raise RuntimeError('Expected exactly one emulator frame')
                    appended = pool.add(state, target, episode, frame) if visible else False
                    score += result.reward
                    hits += int(result.paddle_hit)
                    bricks = int(brick_layout(result.observation).sum())
                    event = outcome(result, bricks)
                    emit(log, dict(episode=episode, frame=frame, state=state.tolist(),
                                   model_state=model_features(state, cfg).tolist(),
                                   executed_target=target, appended_unique=appended,
                                   paddle_x_after=getattr(result, 'paddle_x', None),
                                   target_error_pixels_after=None if getattr(result, 'paddle_x', None) is None
                                   else (target-result.paddle_x)*159,
                                   ball_visible=visible, controller=controller.last,
                                   score=score, hits=hits, bricks=bricks, event=event))
                    if (frame+1) % args.log_every == 0 or event:
                        print(f'[simple-teacher] episode={episode} steps={frame+1} score={score:g} '
                              f'bricks={bricks} hits={hits} unique={len(pool.rows)}', flush=True)
                    if event:
                        break
                episodes.append(dict(episode=episode, seed=start['seed'],
                                     start_paddle_x=start['start_paddle_x'], initial_state=initial.tolist(),
                                     frames=frame+1, score=score, hits=hits,
                                     ended_by=event or 'frame_limit', controller=args.student_controller))
                write_json(folder / 'episodes.json', episodes)
                pool.save(folder, args.rows, cfg)
        report['status'] = 'complete'
    except BaseException as exc:
        report.update(status='error', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        selected = pool.save(folder, args.rows, cfg)
        report.update(requested_rows=args.rows, selected_rows=selected,
                      unique_observed=len(pool.rows), duplicates_skipped=pool.duplicates,
                      requested_rows_reached=args.rows == 0 or selected == args.rows,
                      all_unique_retained=args.rows == 0, completed_episodes=len(episodes))
        write_json(folder / 'summary.json', report)
        print(f'[simple-data] saved={selected} unique rows; requested={args.rows}', flush=True)


def run(args):
    source, cfg, episodes = setup(args)
    include_bricks = False
    read_state = features
    initial_seed = getattr(args, 'initial_seed', None)
    initial_paddle = getattr(args, 'initial_paddle_x', None)
    if (initial_seed is None) != (initial_paddle is None):
        raise ValueError('Specify both --initial-seed and --initial-paddle-x')
    selected = next((r for r in episodes if r['episode'] == args.episode), None)
    if selected is None or (not include_bricks and 'initial_state' not in selected):
        raise ValueError('Use a source collected by ball_paddle collect')
    if selected.get('controller', 'original') != args.student_controller:
        raise ValueError('Student controller must match the collection controller')
    selected = dict(selected)
    if initial_seed is not None:
        if not max(cfg.safe_paddle_margin, cfg.paddle_center_min_x) <= initial_paddle <= min(1-cfg.safe_paddle_margin, cfg.paddle_center_max_x):
            raise ValueError('New paddle initialization lies outside the allowed center range')
        cfg.seed = initial_seed
        selected = dict(seed=initial_seed, start_paddle_x=initial_paddle,
                        controller=args.student_controller, new_initialization=True)
    if args.fresh_model_per_frame and args.backend != 'local':
        raise ValueError('Fresh models require the local backend')
    data_path = args.data or source / ('frame_demonstrations.npz' if include_bricks else 'demonstrations.npz')
    context_mode = getattr(args, 'context_mode', 'fixed')
    selector = None
    if context_mode == 'local':
        if args.backend == 'local' and not args.fresh_model_per_frame:
            raise ValueError('Local GPU context requires --fresh-model-per-frame')
        if args.context_stride != 1:
            raise ValueError('Use --context-stride 1: local selection searches the full saved teacher bank')
    x, y, known, bank_episodes, bank_frames, policy = experiment_inputs.load_bank(
        data_path, args.context_stride, include_bricks)
    if context_mode == 'local':
        selector = LocalTeacherContext(x, bank_episodes, bank_frames,
                                      rows=args.local_context_rows, radius=args.context_window,
                                      motion=getattr(args, 'context_motion', 'compatible'),
                                      velocity_tolerance_pixels=getattr(args, 'velocity_tolerance_pixels', .05),
                                      brick_weight=getattr(args, 'brick_distance_weight', 1.0))
    target_mode = getattr(args, 'prediction_target', 'absolute')
    model_x = experiment_inputs.inputs(x, cfg)
    model_y = model_targets(x, y, target_mode)
    digest = fingerprint(x, y)
    folder = open_unique_run(args.run_dir)
    args.run_dir = str(folder)  # Persist the actual path in experiment.json.
    cfg.save(folder / 'config.json')
    write_json(folder / 'experiment.json', vars(args))
    base_names, model_names = experiment_inputs.names(include_bricks)
    report = dict(status='running', features=base_names, policy_version=policy,
                  model_features=model_names, prediction_target=target_mode,
                  includes_bricks=include_bricks, backend=args.backend,
                  new_client_estimator_per_frame=args.backend == 'client' and selector is not None,
                  controller=args.student_controller,
                  new_initialization=initial_seed is not None,
                  initialization=selected, context_rows=len(x), context_sha256=digest,
                  context_updates=False, teacher_interventions=0, model_instances=0,
                  fit_calls=0, prediction_calls=0, fresh_model_per_frame=args.fresh_model_per_frame,
                  model_settings=model_settings(args))
    report.update(context_mode=context_mode, teacher_bank_rows=len(x),
                  active_context_changes=selector is not None,
                  context_rows=len(x) if selector is None else selector.rows,
                  context_selection=None if selector is None else selector.settings())
    start = perf_counter()
    frames = hits = matches = 0
    score = 0.0
    try:
        with environment(cfg) as env, (folder / 'predictions.jsonl').open('w') as log:
            # Analog collection installs before reset; reproduce the same launch
            # controller for the student, including the requested start position.
            controller = (ControllerTrace(env, args.student_controller)
                          if args.student_controller == 'analog-target' else None)
            env.reset(seed=selected['seed'], start_paddle_x=selected['start_paddle_x'])
            initial = read_state(env)
            if initial_seed is None:
                expected = np.asarray(selected['initial_state'], np.float32)
                if not np.array_equal(initial, expected):
                    raise RuntimeError('Recorded initialization did not reproduce')
            report['initialization'] = dict(selected, initial_state=initial.tolist())
            write_json(folder / 'initialization.json', report['initialization'])
            if controller is None:
                controller = ControllerTrace(env, args.student_controller)
            model = None
            per_frame_client = args.backend == 'client' and selector is not None
            if not args.fresh_model_per_frame and not per_frame_client:
                model, report['model_settings'] = create_model(args)
                model.fit(model_x.copy(), model_y.copy())
                report.update(model_instances=1, fit_calls=1)
            print(f'[simple-student] inputs={model_x.shape[1]} bank_rows={len(x)} context_rows={report["context_rows"]} '
                  f'context_mode={context_mode}', flush=True)
            report['status'] = 'frame_limit'
            for frame in range(args.frames):
                state = read_state(env)
                label = known.get(state.tobytes())
                fit_x, fit_y = model_x, model_y
                query = experiment_inputs.inputs(state, cfg)
                selection = None
                if selector is not None:
                    indices, selection = selector.select(state)
                    fit_x, fit_y = model_x[indices], model_y[indices]
                    selection.update(sha256=fingerprint(fit_x, fit_y),
                                     target_mean=float(fit_y.astype(float).mean()))
                    # A match in the bank does not imply inclusion in this fit.
                    if not np.any(np.all(x[indices] == state, axis=1)):
                        label = None
                matches += int(label is not None)
                inference_start = perf_counter()
                if per_frame_client:
                    raw = predict_client(args, fit_x, fit_y, query, report)
                elif args.fresh_model_per_frame:
                    raw = predict_fresh(args, fit_x, fit_y, query, report)
                else:
                    prediction = np.asarray(model.predict(query[None, :].copy()), float).reshape(-1)
                    if prediction.shape != (1,) or not np.isfinite(prediction).all():
                        raise ValueError('Invalid model output; no fallback used')
                    raw = float(prediction[0])
                    report['prediction_calls'] += 1
                model_output = raw
                prediction_seconds = perf_counter()-inference_start
                raw = destination(model_output, state, target_mode)
                target = env.safe_target_x(raw)
                visible = bool(env.ball_state.visible)
                result = env.step_to_target(target)
                if result.frames_executed != 1:
                    raise RuntimeError('Expected exactly one emulator frame')
                frames += 1
                score += result.reward
                hits += int(result.paddle_hit)
                bricks = int(brick_layout(result.observation).sum())
                event = outcome(result, bricks)
                emit(log, dict(frame=frame, state=state.tolist(), raw_target=raw, target=target,
                               model_state=query.tolist(), model_output=model_output,
                               prediction_seconds=prediction_seconds,
                               ball_visible=visible, prediction_target=target_mode,
                               context_selection=selection,
                               paddle_x_after=getattr(result, 'paddle_x', None),
                               target_error_pixels_after=None if getattr(result, 'paddle_x', None) is None
                               else (target-result.paddle_x)*159,
                               controller=controller.last, exact_context_match=label is not None,
                               context_target=None if label is None else float(label),
                               model_instance=report['model_instances'], score=score,
                               hits=hits, bricks=bricks, event=event))
                if frames % args.log_every == 0 or result.reward or result.paddle_hit or event:
                    print(f'[simple-student] steps={frames} score={score:g} bricks={bricks} '
                          f'hits={hits} context_matches={matches}/{frames} fits={report["fit_calls"]}', flush=True)
                if event:
                    report['status'] = event
                    break
    except BaseException as exc:
        report.update(status='error', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report.update(frames=frames, score=score, hits=hits, exact_context_matches=matches,
                      context_unchanged=digest == fingerprint(x, y),
                      teacher_bank_unchanged=digest == fingerprint(x, y), seconds=perf_counter()-start)
        write_json(folder / 'summary.json', report)


def smoke_test():
    from tempfile import TemporaryDirectory
    from breakout_pfn.ball_paddle_policy import smoke_test as policy_smoke
    policy_smoke()
    from breakout_pfn.wall_features import smoke_test as wall_smoke
    wall_smoke()
    rows = Rows()
    assert rows.add([.2, .3, 0, .01, .5, 0], .2, 0, 0)
    assert not rows.add([.2, .3, 0, .01, .5, 0], .2, 1, 4)
    assert rows.add([.3, .3, 0, .01, .5, 0], .3, 0, 1)
    with TemporaryDirectory() as temporary:
        folder = Path(temporary)
        assert rows.save(folder, 10000) == 2
        x, y, _ = load_context(folder / 'demonstrations.npz', 4)
        assert x.shape == (1, 6) and y.shape == (1,)
    print('Ball/paddle data smoke test passed (no game or model loaded)')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    sub = parser.add_subparsers(dest='command')
    for name in ('collect', 'run'):
        p = sub.add_parser(name)
        p.add_argument('--source', required=True)
        p.add_argument('--run-dir', required=True)
        p.add_argument('--log-every', type=int, default=100)
        p.add_argument('--student-controller', choices=('analog-target',), default='analog-target')
        if name == 'collect':
            p.add_argument('--episodes', type=int, default=3)
            p.add_argument('--episode-frames', type=int, default=20000)
            p.add_argument('--rows', type=int, default=0, help='0 retains every unique row')
            p.add_argument('--collection-seed', type=int, default=10007)
        else:
            p.add_argument('--data')
            p.add_argument('--initial-seed', type=int)
            p.add_argument('--initial-paddle-x', type=float)
            p.add_argument('--episode', type=int, default=0)
            p.add_argument('--frames', type=int, default=20000)
            p.add_argument('--context-stride', type=int, default=1)
            p.add_argument('--velocity-tolerance-pixels', type=float, default=.05,
                           help='Maximum difference per ball velocity component, pixels/frame')
            p.add_argument('--prediction-target', choices=('absolute', 'displacement'), default='absolute')
            p.add_argument('--context-mode', choices=('local',), default='local')
            p.add_argument('--context-motion', choices=('compatible',), default='compatible')
            p.add_argument('--local-context-rows', type=int, default=256)
            p.add_argument('--context-window', type=int, default=8,
                           help='Recorded frames before/after each nearest teacher anchor')
            p.add_argument('--backend', choices=('local', 'client'), default='local')
            p.add_argument('--client-fit-mode', choices=('fit_preprocessors', 'fit_with_cache'), default='fit_preprocessors')
            p.add_argument('--client-n-estimators', type=int, default=1)
            p.add_argument('--fresh-model-per-frame', action='store_true')
            p.add_argument('--model-path', default='v3.5_default')
            p.add_argument('--model-seed', type=int, default=7)
            p.add_argument('--local-fit-mode', choices=('fit_preprocessors', 'fit_with_cache', 'low_memory'), default='fit_preprocessors')
            p.add_argument('--local-n-estimators', type=int, default=1)
            p.add_argument('--local-precision', choices=('auto', 'float32'), default='float32')
            p.add_argument('--gpu-memory-gib', type=float, default=5.0)
            p.add_argument('--gpu-headroom-gib', type=float, default=1.0)
    args = parser.parse_args(argv)
    smoke_test()
    if args.self_test:
        return 0
    if args.command is None:
        parser.error('Choose collect or run')
    if args.command == 'run' and (args.local_context_rows < 2 or args.context_window < 0):
        parser.error('Local context needs at least 2 rows and a nonnegative window radius')
    if hasattr(args, 'rows') and args.rows < 0:
        parser.error('rows must be nonnegative; 0 retains all unique rows')
    for name in ('log_every', 'episodes', 'episode_frames', 'frames', 'context_stride', 'local_n_estimators'):
        if hasattr(args, name) and getattr(args, name) < 1:
            parser.error(f'{name} must be positive')
    (collect if args.command == 'collect' else run)(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
