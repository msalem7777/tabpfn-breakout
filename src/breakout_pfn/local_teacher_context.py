"""Select nearby recorded teacher situations without reading target labels."""
import numpy as np
from breakout_pfn.ball_paddle_policy import validate


class LocalTeacherContext:
    """Nearest state anchors plus short recorded windows in their own episodes.

    Distance uses standardized six-input differences. Ball columns have weight
    one; paddle columns weight 0.25. Direction mismatches add penalties so two
    balls at the same location but moving differently are not treated alike.
    By default, anchors AND window rows must match both velocity signs. Windows
    never include opposite-direction or zero-velocity rows for a moving query.
    The optional mixed mode reproduces the earlier unfiltered experiment.
    An optional component-speed tolerance filters anchors AND their neighbors;
    the game runner enables it by default at 0.05 pixels per frame. It expands
    only when needed to obtain two rows, and reports that expansion explicitly.
    All row indices refer to the immutable, unique teacher bank supplied here.
    """
    def __init__(self, x, episodes, frames, rows=256, radius=8, motion='compatible',
                 velocity_tolerance_pixels=None, brick_weight=1.0):
        full_x = np.asarray(x, np.float32).copy()
        if full_x.ndim != 2 or full_x.shape[1] not in (6, 114):
            raise ValueError('Expected six base inputs, optionally followed by 108 brick cells')
        self.bricks = full_x[:, 6:] if full_x.shape[1] == 114 else None
        self.brick_weight = float(brick_weight)
        if not np.isfinite(self.brick_weight) or self.brick_weight < 0:
            raise ValueError('Brick distance weight must be finite and nonnegative')
        if not np.isfinite(full_x).all():
            raise ValueError('Nonfinite teacher state')
        self.x = full_x[:, :6].copy()
        self.episodes, self.frames = np.asarray(episodes), np.asarray(frames)
        if (self.x.ndim != 2 or self.x.shape[1] != 6 or len(self.x) < 2
                or not np.isfinite(self.x).all()):
            raise ValueError('Local context needs at least two finite six-input rows')
        for metadata in (self.episodes, self.frames):
            if metadata.shape != (len(self.x),) or metadata.dtype.kind not in 'iu' or (metadata < 0).any():
                raise ValueError('Expected nonnegative integer episode/frame metadata per row')
        if rows < 2 or radius < 0:
            raise ValueError('Use at least two context rows and a nonnegative window radius')
        full_x[full_x == 0] = 0
        keys = [row.tobytes() for row in full_x]
        if len(set(keys)) != len(keys):
            raise ValueError('Teacher bank contains duplicate six-input states')
        self.rows, self.radius = min(rows, len(self.x)), radius
        if motion not in ('compatible', 'mixed'):
            raise ValueError('Motion selection must be compatible or mixed')
        self.motion = motion
        if velocity_tolerance_pixels is not None and (not np.isfinite(velocity_tolerance_pixels)
                                                     or velocity_tolerance_pixels < 0):
            raise ValueError('Velocity tolerance must be finite and nonnegative')
        self.velocity_tolerance_pixels = velocity_tolerance_pixels
        self.scale = np.maximum(self.x.astype(float).std(axis=0),
                                [1/159, 1/209, .001, .001, 1/159, .001])
        self.weights = np.asarray([1., 1., 1., 1., .25, .25])
        self.by_episode = {}
        for episode in np.unique(self.episodes):
            ids = np.flatnonzero(self.episodes == episode)
            self.by_episode[int(episode)] = ids[np.argsort(self.frames[ids], kind='stable')]
        self.x.flags.writeable = False

    def select(self, state):
        """Return unique bank indices and auditable selection metadata.

        No live frame number, teacher function, simulated future or target label
        is accepted by this API. Sorting is deterministic, including ties.
        """
        full_state = np.asarray(state, np.float32)
        expected = 114 if self.bricks is not None else 6
        if full_state.shape != (expected,) or not np.isfinite(full_state).all():
            raise ValueError('Query does not match teacher bank inputs')
        state = validate(full_state[:6])
        delta = (self.x.astype(float) - state) / self.scale
        distance = (delta * delta * self.weights).sum(axis=1)
        if self.bricks is not None:
            # Fraction of brick cells that differ. Labels never enter selection.
            distance += self.brick_weight * np.mean(self.bricks != full_state[6:], axis=1)
        distance += 9.0 * (np.sign(self.x[:, 3]) != np.sign(state[3]))
        distance += 4.0 * (np.sign(self.x[:, 2]) != np.sign(state[2]))
        ranked = np.argsort(distance, kind='stable')
        compatible = np.ones(len(self.x), dtype=bool)
        if self.motion == 'compatible':
            # Apply to EVERY row, including temporal neighbors. Zero velocity
            # is its own category. Keep speed differences in the distance rule.
            compatible = ((np.sign(self.x[:, 2]) == np.sign(state[2])) &
                          (np.sign(self.x[:, 3]) == np.sign(state[3])))
        direction_count = int(compatible.sum())
        velocity_error = np.max(abs((self.x[:, 2:4]-state[2:4]) * [159., 209.]), axis=1)
        effective_tolerance = self.velocity_tolerance_pixels
        expanded = False
        if effective_tolerance is not None and direction_count >= 2:
            # Prefer a small speed-matched bank even if fewer than requested
            # rows exist. Only widen enough to get two rows when necessary.
            second_nearest = float(np.partition(velocity_error[compatible], 1)[1])
            effective_tolerance = max(effective_tolerance, second_nearest)
            expanded = effective_tolerance > self.velocity_tolerance_pixels
            compatible &= velocity_error <= effective_tolerance
        ranked = ranked[compatible[ranked]]
        count = min(self.rows, len(ranked))
        if count < 2:
            raise ValueError('Fewer than two motion-compatible teacher rows; no mixed-motion fallback')
        chosen, seen, anchors = [], set(), []
        for anchor in ranked:
            anchor = int(anchor)
            if anchor in seen:
                continue
            anchors.append(anchor)
            episode_ids = self.by_episode[int(self.episodes[anchor])]
            times = self.frames[episode_ids]
            center = int(self.frames[anchor])
            lo = np.searchsorted(times, center-self.radius, side='left')
            hi = np.searchsorted(times, center+self.radius, side='right')
            nearby = episode_ids[lo:hi]
            nearby = nearby[compatible[nearby]]
            # Anchor first, then closest recorded moments before/after it.
            nearby = nearby[np.argsort(abs(self.frames[nearby]-center), kind='stable')]
            for index in [anchor, *nearby.tolist()]:
                index = int(index)
                if index not in seen:
                    chosen.append(index)
                    seen.add(index)
                if len(chosen) == count:
                    break
            if len(chosen) == count:
                break
        # Stable source order avoids unnecessary prompt-order changes when the
        # selected set is unchanged between neighboring student states.
        indices = np.asarray(sorted(chosen), np.int64)
        return indices, dict(rows=len(indices), compatible_available=len(ranked),
                             direction_compatible_available=direction_count,
                             velocity_tolerance_pixels=effective_tolerance,
                             velocity_tolerance_expanded=expanded,
                             maximum_velocity_error_pixels=float(velocity_error[indices].max()),
                             motion_filter=self.motion, anchor_indices=anchors,
                             bank_indices=indices.tolist(),
                             episodes=self.episodes[indices].tolist(),
                             frames=self.frames[indices].tolist(),
                             nearest_distance=float(distance[ranked[0]]),
                             maximum_selected_distance=float(distance[indices].max()))

    def settings(self):
        return dict(rows=self.rows, window_radius_frames=self.radius,
                    includes_bricks=self.bricks is not None, brick_distance_weight=self.brick_weight,
                    velocity_tolerance_pixels=self.velocity_tolerance_pixels,
                    motion_filter=self.motion,
                    scales=self.scale.tolist(), squared_distance_weights=self.weights.tolist(),
                    vertical_direction_penalty=9.0, horizontal_direction_penalty=4.0)


def smoke_test():
    x = np.zeros((20, 6), np.float32)
    x[:, 0] = np.arange(20)/20
    x[:, 3] = .01
    selector = LocalTeacherContext(x, np.zeros(20, np.int64), np.arange(20), rows=5, radius=2)
    ids, _ = selector.select(x[10])
    assert ids.tolist() == [8, 9, 10, 11, 12]
    assert np.array_equal(ids, selector.select(x[10])[0])
    print('Local teacher context smoke test passed (no GPU/model loaded)')


if __name__ == '__main__':
    smoke_test()
